import json
import torch
import os
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoConfig, AutoTokenizer
from herference.model import S2E
from pathlib import Path
import argparse
import csv

# ============================================================
# KONFIGURACJA
# ============================================================

MODEL_NAME = "ipipan/herference-large"

TRAIN_PATH = "train"
VAL_PATH = "val"
VERSION = "v2"

BATCH_SIZE = 1
LEARNING_RATE = 5e-6
EPOCHS = 10

MAX_MODEL_INPUT = 512
MAX_SPAN_LENGTH = 30

CHECKPOINT_EPOCH = 5

OUTPUT_DIR = "./finetuned_herference"
CHECKPOINT_DIR = "./checkpoints"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# PARAMETRY HERFERENCE
# ============================================================

class ModelParams:
    max_span_length = MAX_SPAN_LENGTH
    top_lambda = 0.4
    ffnn_size = 3072
    normalise_loss = True
    max_model_input_length = MAX_MODEL_INPUT
    dropout_prob = 0.3
    null_id_for_coref = 0


# ============================================================
# DATASET
# ============================================================

class CorefDataset(Dataset):

    def __init__(self, path, tokenizer):

        self.tokenizer = tokenizer
        self.data = []

        if os.path.isdir(path):

            files = sorted(
                os.path.join(path, filename)
                for filename in os.listdir(path)
                if filename.endswith(".jsonl")
            )

            if not files:
                raise RuntimeError(
                    f"Nie znaleziono plików .jsonl w folderze: {path}"
                )

        elif os.path.isfile(path):

            files = [path]

        else:

            raise RuntimeError(
                f"Nie znaleziono ścieżki: {path}"
            )

        # ----------------------------------------------------
        # Wczytanie wszystkich dokumentów
        # ----------------------------------------------------

        for file_path in files:

            with open(
                    file_path,
                    "r",
                    encoding="utf-8"
            ) as f:

                for line in f:

                    line = line.strip()

                    if not line:
                        continue

                    item = json.loads(line)

                    item["_source_file"] = file_path

                    self.data.append(item)

        print()
        print("=" * 60)
        print("DATASET")
        print("=" * 60)
        print("Ścieżka:", path)
        print("Plików:", len(files))
        print("Dokumentów:", len(self.data))

        for file_path in files:
            print(" ", file_path)

        print("=" * 60)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        item = self.data[idx]

        words = [
            token
            for sentence in item["sentences"]
            for token in sentence
        ]

        # ----------------------------------------------------
        # Tokenizacja
        # ----------------------------------------------------

        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            return_tensors="pt",
            padding=False,
            truncation=False
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        word_ids = encoding.word_ids(
            batch_index=0
        )

        # ----------------------------------------------------
        # WORD -> TOKEN
        # ----------------------------------------------------

        word_to_tokens = {}

        for token_idx, word_idx in enumerate(word_ids):

            if word_idx is None:
                continue

            if word_idx not in word_to_tokens:
                word_to_tokens[word_idx] = []

            word_to_tokens[word_idx].append(
                token_idx
            )

        # ----------------------------------------------------
        # GOLD CLUSTERS
        # ----------------------------------------------------

        token_clusters = []

        for cluster in item.get("clusters", []):

            token_cluster = []

            for word_start, word_end in cluster:

                if word_start not in word_to_tokens:
                    continue

                if word_end not in word_to_tokens:
                    continue

                token_start = min(
                    word_to_tokens[word_start]
                )

                token_end = max(
                    word_to_tokens[word_end]
                )

                token_cluster.append(
                    [token_start, token_end]
                )

            if token_cluster:
                token_clusters.append(
                    token_cluster
                )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_clusters": token_clusters,
            "doc_key": item.get(
                "doc_key",
                f"document_{idx}"
            ),
            "source_file": item.get(
                "_source_file",
                ""
            ),
            "num_words": len(words),
            "num_tokens": len(input_ids)
        }


# ============================================================
# COLLATE
# ============================================================

def collate_fn(batch):
    # --------------------------------------------------------
    # batch_size = 1
    # --------------------------------------------------------

    if len(batch) != 1:
        raise RuntimeError(
            "Ten collate_fn obsługuje obecnie "
            "batch_size=1."
        )

    item = batch[0]

    input_ids = item["input_ids"].unsqueeze(0)

    attention_mask = (
        item["attention_mask"]
        .unsqueeze(0)
    )

    token_clusters = item["token_clusters"]

    # --------------------------------------------------------
    # GOLD CLUSTERS
    # --------------------------------------------------------

    if not token_clusters:

        padded = torch.zeros(
            (1, 1, 1, 2),
            dtype=torch.long
        )

    else:

        max_clusters = len(token_clusters)

        max_mentions = max(
            len(cluster)
            for cluster in token_clusters
        )

        padded = torch.zeros(
            (
                1,
                max_clusters,
                max_mentions,
                2
            ),
            dtype=torch.long
        )

        for c, cluster in enumerate(
                token_clusters
        ):

            for m, (start, end) in enumerate(
                    cluster
            ):
                padded[
                    0, c, m, 0
                ] = start

                padded[
                    0, c, m, 1
                ] = end

    return {
        "input_ids": input_ids.to(DEVICE),
        "attention_mask": attention_mask.to(DEVICE),
        "gold_clusters": padded.to(DEVICE),

        "doc_key": item["doc_key"],
        "source_file": item["source_file"],
        "num_words": item["num_words"],
        "num_tokens": item["num_tokens"]
    }

def get_experiment_name():
    parser = argparse.ArgumentParser(
        description="Uczenie modelu herference dla wybranego eksperymentu."
    )

    parser.add_argument(
        "experiment",
        type=str,
        help="Nazwa eksperymentu, np. experiment_1"
    )

    args = parser.parse_args()

    return args.experiment


def load_experiment_config(
        documents_config_path,
        experiments_config_path,
        experiment_name
):
    """
    Wczytuje konfigurację dokumentów oraz wybrany eksperyment.

    Zwraca:
        {
            "train": [...pełne ścieżki...],
            "validation": [...pełne ścieżki...],
            "test": [...pełne ścieżki...]
        }
    """

    with open(documents_config_path, "r", encoding="utf-8") as f:
        documents = json.load(f)

    with open(experiments_config_path, "r", encoding="utf-8") as f:
        experiments = json.load(f)

    if experiment_name not in experiments:
        available = ", ".join(experiments.keys())
        raise ValueError(
            f"Nie znaleziono eksperymentu '{experiment_name}'. "
            f"Dostępne konfiguracje: {available}"
        )

    experiment = experiments[experiment_name]

    result = {}

    for split in ["train", "validation", "test"]:
        result[split] = []

        for document_name in experiment[split]:

            if document_name not in documents:
                raise ValueError(
                    f"Dokument '{document_name}' nie został "
                    f"zdefiniowany w {documents_config_path}"
                )

            path = Path(documents[document_name])

            if not path.exists():
                raise FileNotFoundError(
                    f"Nie znaleziono pliku dokumentu: {path}"
                )

            result[split].append(str(path))

    return result


# ============================================================
# TRENING
# ============================================================

def run_training():
    # ============================================================
    # KONFIGURACJA EKSPERYMENTU
    # ============================================================

    experiment_name = get_experiment_name()

    DOCUMENTS_CONFIG = "config/documents.json"
    EXPERIMENTS_CONFIG = "config/experiments.json"

    experiment = load_experiment_config(
        DOCUMENTS_CONFIG,
        EXPERIMENTS_CONFIG,
        experiment_name
    )

    train_paths = experiment["train"]
    val_paths = experiment["validation"]

    # ------------------------------------------------------------
    # Katalog wyników
    # ------------------------------------------------------------

    OUTPUT_ROOT = "finetuned_herference"

    output_dir = os.path.join(
        OUTPUT_ROOT,
        experiment_name + VERSION
    )

    file_name = experiment_name + VERSION + ".csv"

    loss_csv_path = os.path.join("training", file_name)

    with open(loss_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss"])

    max_epochs = EPOCHS
    patience = 2

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    # ------------------------------------------------------------
    # PARAMETRY SLIDING WINDOW
    # ------------------------------------------------------------

    WINDOW_SIZE = 400
    WINDOW_OVERLAP = 200
    WINDOW_STEP = WINDOW_SIZE - WINDOW_OVERLAP

    # ============================================================
    # INFORMACJE O EKSPERYMENCIE
    # ============================================================

    print("\n" + "=" * 70)
    print("HERFERENCE - FINE-TUNING")
    print("=" * 70)

    print(
        f"EKSPERYMENT:       {experiment_name}"
    )

    print()
    print("TRAIN:")

    for path in train_paths:
        print(
            f"  - {path}"
        )

    print()
    print("VAL:")

    for path in val_paths:
        print(
            f"  - {path}"
        )

    print()
    print(f"OUTPUT:            {output_dir}")
    print(f"MAX EPOCHS:        {max_epochs}")
    print(f"EARLY STOPPING:    patience={patience}")
    print(f"BATCH SIZE:        1")
    print(f"LR:                {LEARNING_RATE}")
    print(f"DEVICE:            {DEVICE}")
    print(f"MAX MODEL INPUT:   {MAX_MODEL_INPUT}")
    print(f"WINDOW SIZE:       {WINDOW_SIZE}")
    print(f"WINDOW OVERLAP:    {WINDOW_OVERLAP}")
    print("CHECKPOINT:        tylko najlepsza epoka")

    # ============================================================
    # CUDA
    # ============================================================

    if torch.cuda.is_available():

        print()
        print("CUDA:")

        print(
            f"  GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

        print(
            f"  VRAM: "
            f"{torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB"
        )

        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    # ============================================================
    # TOKENIZER + CONFIG
    # ============================================================

    print("\nŁadowanie konfiguracji modelu...")

    config = AutoConfig.from_pretrained(
        MODEL_NAME
    )

    print("Ładowanie tokenizera...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    params = ModelParams()

    # ============================================================
    # MODEL
    # ============================================================

    print("\nŁadowanie modelu...")

    model = S2E.from_pretrained(
        MODEL_NAME,
        config=config,
        params=params
    ).to(DEVICE)

    # ------------------------------------------------------------
    # Gradient checkpointing
    # ------------------------------------------------------------

    try:

        model.bert.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={
                "use_reentrant": False
            }
        )

        print(
            "Gradient checkpointing: "
            "włączony (non-reentrant)"
        )

    except TypeError:

        model.bert.gradient_checkpointing_enable()

        print(
            "Gradient checkpointing: "
            "włączony"
        )

    print("Model załadowany.")

    print(
        f"Parametry: "
        f"{sum(p.numel() for p in model.parameters()):,}"
    )

    # ============================================================
    # DATASETS
    # ============================================================

    # ------------------------------------------------------------
    # TRAIN
    # ------------------------------------------------------------

    print("\nŁadowanie TRAIN...")

    train_datasets = []

    for path in train_paths:

        dataset = CorefDataset(
            path,
            tokenizer
        )

        train_datasets.append(
            dataset
        )

    train_dataset = torch.utils.data.ConcatDataset(
        train_datasets
    )

    print(
        f"TRAIN documents: "
        f"{len(train_dataset)}"
    )

    # ------------------------------------------------------------
    # VALIDATION
    # ------------------------------------------------------------

    print("\nŁadowanie VAL...")

    val_datasets = []

    for path in val_paths:

        dataset = CorefDataset(
            path,
            tokenizer
        )

        val_datasets.append(
            dataset
        )

    val_dataset = torch.utils.data.ConcatDataset(
        val_datasets
    )

    print(
        f"VAL documents: "
        f"{len(val_dataset)}"
    )

    # ============================================================
    # DATALOADER
    # ============================================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )

    # ============================================================
    # GOLD CLUSTERS DLA OKNA
    # ============================================================

    def create_window_clusters(
            gold_clusters,
            window_start,
            window_end
    ):

        """
        Przekształca globalne indeksy tokenów
        na indeksy lokalne dla danego okna.

        Zachowywane są wyłącznie wzmianki,
        które w całości znajdują się w oknie.
        """

        window_clusters = []

        if gold_clusters is None:
            return window_clusters

        clusters = gold_clusters[0]

        for cluster in clusters:

            mentions = []

            for mention in cluster:

                start = int(
                    mention[0].item()
                )

                end = int(
                    mention[1].item()
                )

                # ------------------------------------------------
                # Padding [0,0]
                # ------------------------------------------------

                if start == 0 and end == 0:
                    continue

                # ------------------------------------------------
                # Wzmianka musi znajdować się
                # w całości w aktualnym oknie.
                # ------------------------------------------------

                if start < window_start:
                    continue

                if end >= window_end:
                    continue

                local_start = (
                    start - window_start
                )

                local_end = (
                    end - window_start
                )

                mentions.append(
                    [
                        local_start,
                        local_end
                    ]
                )

            if mentions:

                window_clusters.append(
                    mentions
                )

        return window_clusters

    # ============================================================
    # PRZYGOTOWANIE GOLD TENSOR
    # ============================================================

    def clusters_to_tensor(
            clusters
    ):

        """
        Zamienia listę klastrów:
            [
                [[start, end], ...],
                ...
            ]

        na tensor:
            [1, clusters, mentions, 2]
        """

        if not clusters:

            return torch.zeros(
                (1, 1, 1, 2),
                dtype=torch.long,
                device=DEVICE
            )

        max_clusters = len(
            clusters
        )

        max_mentions = max(
            len(cluster)
            for cluster in clusters
        )

        tensor = torch.zeros(
            (
                1,
                max_clusters,
                max_mentions,
                2
            ),
            dtype=torch.long,
            device=DEVICE
        )

        for c, cluster in enumerate(
                clusters
        ):

            for m, (start, end) in enumerate(
                    cluster
            ):

                tensor[
                    0, c, m, 0
                ] = start

                tensor[
                    0, c, m, 1
                ] = end

        return tensor

    # ============================================================
    # WYZNACZENIE OKIEN
    # ============================================================

    def get_windows(
            num_tokens
    ):

        windows = []

        start = 0

        while start < num_tokens:

            end = min(
                start + WINDOW_SIZE,
                num_tokens
            )

            windows.append(
                (start, end)
            )

            if end >= num_tokens:
                break

            start += WINDOW_STEP

        return windows

    # ============================================================
    # UCZENIE JEDNEGO DOKUMENTU
    # ============================================================

    def train_document(
            model,
            batch,
            optimizer,
            document_name
    ):

        input_ids = batch[
            "input_ids"
        ]

        attention_mask = batch[
            "attention_mask"
        ]

        gold_clusters = batch[
            "gold_clusters"
        ]

        total_tokens = input_ids.shape[1]

        windows = get_windows(
            total_tokens
        )

        print(
            f"SLIDING WINDOW: "
            f"chunk={WINDOW_SIZE}, "
            f"overlap={WINDOW_OVERLAP}"
        )

        print(
            f"LICZBA OKIEN: "
            f"{len(windows)}"
        )

        document_loss = 0.0
        valid_windows = 0

        for window_idx, (
                window_start,
                window_end
        ) in enumerate(windows):

            optimizer.zero_grad(
                set_to_none=True
            )

            window_input_ids = (
                input_ids[
                    :,
                    window_start:window_end
                ]
            )

            window_attention_mask = (
                attention_mask[
                    :,
                    window_start:window_end
                ]
            )

            local_clusters = (
                create_window_clusters(
                    gold_clusters,
                    window_start,
                    window_end
                )
            )

            local_gold_clusters = (
                clusters_to_tensor(
                    local_clusters
                )
            )

            if window_idx == 0:

                print(
                    f"INPUT: "
                    f"{window_input_ids.shape}"
                )

                print(
                    f"ATTENTION: "
                    f"{window_attention_mask.shape}"
                )

            try:

                outputs = model(
                    input_ids=window_input_ids,
                    attention_mask=window_attention_mask,
                    gold_clusters=local_gold_clusters,
                    debug_coref=None
                )

                loss = outputs[0]

                if not torch.isfinite(loss):

                    print(
                        f"WARNING: "
                        f"nieprawidłowy loss "
                        f"w oknie "
                        f"{window_idx + 1}: "
                        f"{loss.item()}"
                    )

                    del outputs
                    del loss

                    optimizer.zero_grad(
                        set_to_none=True
                    )

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    continue

                loss_value = loss.item()

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0
                )

                optimizer.step()

                document_loss += loss_value
                valid_windows += 1

                del outputs
                del loss
                del window_input_ids
                del window_attention_mask
                del local_gold_clusters

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except torch.cuda.OutOfMemoryError:

                print()
                print(
                    "!!! CUDA OUT OF MEMORY !!!"
                )

                print(
                    f"Dokument: "
                    f"{document_name}"
                )

                print(
                    f"Window: "
                    f"{window_idx + 1}/"
                    f"{len(windows)}"
                )

                print(
                    f"Zakres: "
                    f"{window_start}:{window_end}"
                )

                optimizer.zero_grad(
                    set_to_none=True
                )

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                print(
                    "Pomijam to okno "
                    "i przechodzę dalej."
                )

                continue

        # ========================================================
        # DOCUMENT LOSS
        # ========================================================

        if valid_windows > 0:

            avg_document_loss = (
                document_loss /
                valid_windows
            )

        else:

            avg_document_loss = float(
                "nan"
            )

        print()
        print(
            f"DOCUMENT LOSS: "
            f"{avg_document_loss:.6f}"
        )

        return avg_document_loss


    def validate_document(
            model,
            batch,
            document_name
    ):

        input_ids = batch[
            "input_ids"
        ]

        attention_mask = batch[
            "attention_mask"
        ]

        gold_clusters = batch[
            "gold_clusters"
        ]

        total_tokens = input_ids.shape[1]

        windows = get_windows(
            total_tokens
        )

        document_loss = 0.0
        valid_windows = 0

        print(
            f"VALIDATION SLIDING WINDOW: "
            f"{len(windows)} okien"
        )

        with torch.no_grad():

            for window_idx, (
                    window_start,
                    window_end
            ) in enumerate(windows):

                window_input_ids = (
                    input_ids[
                        :,
                        window_start:window_end
                    ]
                )

                window_attention_mask = (
                    attention_mask[
                        :,
                        window_start:window_end
                    ]
                )

                local_clusters = (
                    create_window_clusters(
                        gold_clusters,
                        window_start,
                        window_end
                    )
                )

                local_gold_clusters = (
                    clusters_to_tensor(
                        local_clusters
                    )
                )

                try:

                    outputs = model(
                        input_ids=window_input_ids,
                        attention_mask=window_attention_mask,
                        gold_clusters=local_gold_clusters,
                        debug_coref=None
                    )

                    loss = outputs[0]

                    if not torch.isfinite(loss):

                        print(
                            "WARNING: "
                            "nieprawidłowy "
                            "validation loss"
                        )

                        del outputs
                        del loss

                        continue

                    loss_value = loss.item()

                    document_loss += loss_value
                    valid_windows += 1

                    del outputs
                    del loss
                    del window_input_ids
                    del window_attention_mask
                    del local_gold_clusters

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                except torch.cuda.OutOfMemoryError:

                    print(
                        "WARNING: "
                        "CUDA OOM podczas walidacji. "
                        "Pomijam okno."
                    )

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    continue

        if valid_windows > 0:

            return (
                document_loss /
                valid_windows
            )

        return float("nan")

    # ============================================================
    # OPTIMIZER
    # ============================================================

    no_decay = [
        "bias",
        "LayerNorm.weight"
    ]

    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(
                    nd in n
                    for nd in no_decay
                )
            ],
            "weight_decay": 0.01
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(
                    nd in n
                    for nd in no_decay
                )
            ],
            "weight_decay": 0.0
        }
    ]

    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=LEARNING_RATE
    )

    print()
    print(
        "Optimizer: AdamW"
    )

    # ============================================================
    # EARLY STOPPING
    # ============================================================

    best_val_loss = float("inf")
    best_epoch = None
    epochs_without_improvement = 0

    # ============================================================
    # UCZENIE
    # ============================================================

    for epoch in range(max_epochs):

        print("\n")
        print("=" * 70)

        print(
            f"EPOCH "
            f"{epoch + 1}/{max_epochs}"
        )

        print("=" * 70)

        # ========================================================
        # TRAIN
        # ========================================================

        model.train()

        total_train_loss = 0.0
        train_documents = 0

        print("\n--- TRAIN ---")

        for step, batch in enumerate(
                train_loader
        ):

            document_name = batch[
                "doc_key"
            ]

            print()
            print("=" * 70)

            print(
                f"TRAIN | Dokument: "
                f"{document_name}"
            )

            print(
                f"Źródło: "
                f"{batch['source_file']}"
            )

            print(
                f"Liczba tokenów: "
                f"{batch['num_tokens']}"
            )

            print(
                f"Liczba słów: "
                f"{batch['num_words']}"
            )

            print("=" * 70)

            loss_value = train_document(
                model=model,
                batch=batch,
                optimizer=optimizer,
                document_name=document_name
            )

            if torch.isfinite(
                    torch.tensor(loss_value)
            ):

                total_train_loss += (
                    loss_value
                )

                train_documents += 1

            if torch.cuda.is_available():

                torch.cuda.empty_cache()

                print(
                    f"GPU end document: "
                    f"{torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB"
                )

        # ========================================================
        # TRAIN LOSS
        # ========================================================

        if train_documents > 0:

            avg_train_loss = (
                total_train_loss /
                train_documents
            )

        else:

            avg_train_loss = float("nan")

        print()
        print("-" * 70)

        print(
            f"TRAIN LOSS: "
            f"{avg_train_loss:.6f}"
        )

        print("-" * 70)

        # ========================================================
        # VALIDATION
        # ========================================================

        model.eval()

        total_val_loss = 0.0
        val_documents = 0

        print("\n--- VALIDATION ---")

        for val_step, batch in enumerate(
                val_loader
        ):

            document_name = batch[
                "doc_key"
            ]

            print()
            print("=" * 70)

            print(
                f"VAL | Dokument: "
                f"{document_name}"
            )

            print(
                f"Liczba tokenów: "
                f"{batch['num_tokens']}"
            )

            print("=" * 70)

            val_document_loss = (
                validate_document(
                    model=model,
                    batch=batch,
                    document_name=document_name
                )
            )

            if torch.isfinite(
                    torch.tensor(
                        val_document_loss
                    )
            ):

                total_val_loss += (
                    val_document_loss
                )

                val_documents += 1

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # ========================================================
        # VALIDATION LOSS
        # ========================================================

        if val_documents > 0:

            avg_val_loss = (
                total_val_loss /
                val_documents
            )

        else:

            avg_val_loss = float("nan")

        # ========================================================
        # EPOCH SUMMARY
        # ========================================================

        with open(loss_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                avg_train_loss,
                avg_val_loss
            ])

        print("\n")
        print("=" * 70)

        print(
            f"EPOCH "
            f"{epoch + 1}/{max_epochs} "
            f"PODSUMOWANIE"
        )

        print("=" * 70)

        print(
            f"TRAIN LOSS: "
            f"{avg_train_loss:.6f}"
        )

        print(
            f"VAL LOSS:   "
            f"{avg_val_loss:.6f}"
        )

        # ========================================================
        # BEST MODEL
        # ========================================================

        improved = (
            torch.isfinite(
                torch.tensor(avg_val_loss)
            )
            and avg_val_loss < best_val_loss
        )

        if improved:

            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            epochs_without_improvement = 0

            best_dir = os.path.join(
                output_dir,
                "best"
            )

            os.makedirs(
                best_dir,
                exist_ok=True
            )

            print()
            print(
                "Nowy najlepszy model!"
            )

            print(
                f"Epoch: "
                f"{best_epoch}"
            )

            print(
                f"Validation loss: "
                f"{best_val_loss:.6f}"
            )

            # ----------------------------------------------------
            # Model
            # ----------------------------------------------------

            model.save_pretrained(
                best_dir
            )

            # ----------------------------------------------------
            # Tokenizer
            # ----------------------------------------------------

            tokenizer.save_pretrained(
                best_dir
            )

            # ----------------------------------------------------
            # State dict
            # ----------------------------------------------------

            torch.save(
                model.state_dict(),
                os.path.join(
                    best_dir,
                    "pytorch_model.bin"
                )
            )

            # ----------------------------------------------------
            # Model params
            # ----------------------------------------------------

            torch.save(
                vars(params),
                os.path.join(
                    best_dir,
                    "model_params.pt"
                )
            )

            # ----------------------------------------------------
            # Training state
            # ----------------------------------------------------

            torch.save(
                {
                    "experiment_name":
                        experiment_name,

                    "train_documents":
                        train_paths,

                    "validation_documents":
                        val_paths,

                    "epoch":
                        best_epoch,

                    "train_loss":
                        avg_train_loss,

                    "val_loss":
                        avg_val_loss,

                    "best_val_loss":
                        best_val_loss,

                    "model_name":
                        MODEL_NAME,

                    "max_model_input":
                        MAX_MODEL_INPUT,

                    "max_span_length":
                        MAX_SPAN_LENGTH,

                    "learning_rate":
                        LEARNING_RATE,

                    "window_size":
                        WINDOW_SIZE,

                    "window_overlap":
                        WINDOW_OVERLAP,

                    "max_epochs":
                        max_epochs,

                    "patience":
                        patience,

                    "optimizer_state_dict":
                        optimizer.state_dict(),
                },
                os.path.join(
                    best_dir,
                    "training_state.pt"
                )
            )

            print(
                f"Najlepszy model zapisany: "
                f"{best_dir}"
            )

        else:

            epochs_without_improvement += 1

            print()
            print(
                "Brak poprawy validation loss."
            )

            print(
                f"Brak poprawy przez: "
                f"{epochs_without_improvement}/"
                f"{patience} epoki."
            )

        # ========================================================
        # EARLY STOPPING
        # ========================================================

        if epochs_without_improvement >= patience:

            print()
            print("=" * 70)

            print(
                "EARLY STOPPING"
            )

            print(
                f"Brak poprawy validation loss "
                f"przez {patience} kolejne epoki."
            )

            print(
                f"Najlepsza epoka: "
                f"{best_epoch}"
            )

            print(
                f"Najlepszy validation loss: "
                f"{best_val_loss:.6f}"
            )

            print("=" * 70)

            break

        # ========================================================
        # CLEANUP EPOCH
        # ========================================================

        if torch.cuda.is_available():

            torch.cuda.empty_cache()

            print(
                f"\nGPU po epoce: "
                f"{torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB"
            )

        model.train()

    # ============================================================
    # KONIEC UCZENIA
    # ============================================================

    print("\n")
    print("=" * 70)

    print(
        "UCZENIE ZAKOŃCZONE"
    )

    print("=" * 70)

    print(
        f"Eksperyment: "
        f"{experiment_name}"
    )

    print(
        f"Najlepsza epoka: "
        f"{best_epoch}"
    )

    print(
        f"Najlepszy validation loss: "
        f"{best_val_loss:.6f}"
    )

    print(
        f"Model zapisany wyłącznie w: "
        f"{os.path.join(output_dir, 'best')}"
    )

    print()
    print(
        "Dokument testowy nie był używany "
        "podczas uczenia ani walidacji."
    )

    # ============================================================
    # FINAL CUDA CLEANUP
    # ============================================================

    if torch.cuda.is_available():

        torch.cuda.empty_cache()

        print(
            f"GPU po uczeniu: "
            f"{torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB"
        )

    print()
    print("=" * 70)

    print(
        "GOTOWE"
    )

    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    run_training()
