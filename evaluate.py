import csv
import json
import os

import torch
from transformers import AutoConfig, AutoTokenizer

from conll import input_ids_to_tokens, write_conll_file
from herference.model import S2E
from train_base import load_experiment_config, get_experiment_name

# ============================================================
# KONFIGURACJA
# ============================================================

MODEL_NAME = "ipipan/herference-large"

EXPERIMENT_NAME = get_experiment_name()

DOCUMENTS_CONFIG = "config/documents.json"
EXPERIMENTS_CONFIG = "config/experiments.json"

experiment = load_experiment_config(
    DOCUMENTS_CONFIG,
    EXPERIMENTS_CONFIG,
    EXPERIMENT_NAME
)

TEST_PATH = experiment["test"][0]

VERSION = "v3"

print(TEST_PATH)

FINETUNED_MODEL = os.path.join(
    "finetuned_herference",
    EXPERIMENT_NAME + VERSION,
    "best"
)

# ------------------------------------------------------------
# PARAMETRY
# ------------------------------------------------------------

MAX_MODEL_INPUT = 512
MAX_SPAN_LENGTH = 30

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
# WCZYTANIE DOKUMENTU TESTOWEGO
# ============================================================

def load_test_document():
    if os.path.isdir(TEST_PATH):

        files = sorted(
            os.path.join(TEST_PATH, filename)
            for filename in os.listdir(TEST_PATH)
            if filename.endswith(".jsonl")
        )

    elif os.path.isfile(TEST_PATH):

        files = [TEST_PATH]

    else:

        raise RuntimeError(
            f"Nie znaleziono TEST_PATH: {TEST_PATH}"
        )

    if not files:
        raise RuntimeError(
            f"Nie znaleziono plików .jsonl w: {TEST_PATH}"
        )

    if len(files) != 1:
        raise RuntimeError(
            "\n"
            "TEST powinien zawierać dokładnie 1 dokument.\n"
            f"Znaleziono plików: {len(files)}\n"
            f"{files}"
        )

    test_file = files[0]

    print()
    print("=" * 70)
    print("DOKUMENT TESTOWY")
    print("=" * 70)
    print("Plik:", test_file)

    documents = []

    with open(
            test_file,
            "r",
            encoding="utf-8"
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            documents.append(
                json.loads(line)
            )

    if len(documents) != 1:
        raise RuntimeError(
            "\n"
            "Plik testowy powinien zawierać dokładnie jeden dokument.\n"
            f"Znaleziono: {len(documents)}"
        )

    document = documents[0]

    print(
        "Doc key:",
        document.get("doc_key", "brak")
    )

    print(
        "Liczba zdań:",
        len(document["sentences"])
    )

    print(
        "Liczba gold clusters:",
        len(document.get("clusters", []))
    )

    print("=" * 70)

    return document


# ============================================================
# TOKENIZACJA
# ============================================================

def tokenize_document(document, tokenizer):
    words = [
        token
        for sentence in document["sentences"]
        for token in sentence
    ]

    encoding = tokenizer(
        words,
        is_split_into_words=True,
        return_tensors="pt",
        padding=False,
        truncation=False
    )

    input_ids = encoding["input_ids"].squeeze(0)

    attention_mask = (
        encoding["attention_mask"].squeeze(0)
    )

    word_ids = encoding.word_ids(
        batch_index=0
    )

    # --------------------------------------------------------
    # WORD -> TOKEN
    # --------------------------------------------------------

    word_to_tokens = {}

    for token_idx, word_idx in enumerate(word_ids):

        if word_idx is None:
            continue

        if word_idx not in word_to_tokens:
            word_to_tokens[word_idx] = []

        word_to_tokens[word_idx].append(
            token_idx
        )

    return (
        words,
        input_ids,
        attention_mask,
        word_to_tokens
    )


# ============================================================
# GOLD CLUSTERS
#
# Zamiana:
#
# word indices
#       ↓
# token indices
# ============================================================

def convert_gold_clusters(
        document,
        word_to_tokens
):
    gold_clusters = []

    for cluster in document.get(
            "clusters",
            []
    ):

        token_cluster = []

        for mention in cluster:

            word_start = int(
                mention[0]
            )

            word_end = int(
                mention[1]
            )

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
                (
                    token_start,
                    token_end
                )
            )

        if token_cluster:
            gold_clusters.append(
                token_cluster
            )

    return gold_clusters


# ============================================================
# MODEL
# ============================================================

def load_model(model_path):
    print()
    print("=" * 70)
    print("ŁADOWANIE MODELU")
    print("=" * 70)

    print(
        "Model:",
        model_path
    )

    config = AutoConfig.from_pretrained(
        model_path
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path
    )

    params = ModelParams()

    model = S2E.from_pretrained(
        model_path,
        config=config,
        params=params
    )

    model = model.to(DEVICE)

    model.eval()

    print(
        "Model załadowany."
    )

    print(
        "Parametry:",
        f"{sum(p.numel() for p in model.parameters()):,}"
    )

    print("=" * 70)

    return model, tokenizer


# ============================================================
# PREDYKCJA
# ============================================================

def aggregate_antecedent_votes(
        mention_antecedent_votes
):
    """
    Dla każdego mentionu wybiera poprzednika
    z najwyższym score spośród wszystkich okien.
    """

    mention_to_antecedent = {}

    for mention, votes in mention_antecedent_votes.items():

        if not votes:
            continue

        selected = max(
            votes,
            key=lambda x: x["score"]
        )

        mention_to_antecedent[mention] = selected["antecedent"]

    return mention_to_antecedent


def predict(model, input_ids, attention_mask, device, gold_clusters=None):
    import torch

    from collections import defaultdict

    mention_antecedent_votes = defaultdict(list)
    mention_windows = defaultdict(set)

    model.eval()

    # 1. Przeniesienie danych na właściwe urządzenie

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    # 2. Predykcja

    with torch.no_grad():

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_all_outputs=True,
            debug_coref=False
        )

    # 3. Sprawdzenie struktury outputs

    print()
    print("Typ outputs:", type(outputs))
    print(
        "Liczba elementów outputs:",
        len(outputs) if hasattr(outputs, "__len__") else "brak"
    )

    if not isinstance(outputs, list):
        raise RuntimeError(
            "Nieoczekiwana struktura outputs: "
            f"oczekiwano list, otrzymano {type(outputs)}"
        )

    if len(outputs) != 1:
        raise RuntimeError(
            "Nieoczekiwana liczba elementów outputs: "
            f"{len(outputs)}"
        )

    # 4. Outputs dla dokumentu

    document_outputs = outputs[0]

    if not isinstance(document_outputs, list):
        raise RuntimeError(
            "Nieoczekiwana struktura document_outputs: "
            f"{type(document_outputs)}"
        )

    print(
        "Liczba okien:",
        len(document_outputs)
    )

    # 5. Odtworzenie dokładnie tych samych okien

    try:
        from herference.model import split_with_overlap
    except ImportError:
        raise ImportError(
            "Nie udało się zaimportować split_with_overlap "
            "z herference.model."
        )

    _, spans = split_with_overlap(
        input_ids[0],
        chunk_size=400,
        overlap=200,
        min_chunk_size=50
    )

    if len(spans) != len(document_outputs):
        raise RuntimeError(
            "Liczba odtworzonych okien nie zgadza się "
            "z liczbą okien zwróconych przez model.\n"
            f"Model: {len(document_outputs)}\n"
            f"Odtworzone: {len(spans)}"
        )

    # 6. Struktury do budowania klastrów

    # mention_to_antecedent = {}

    all_mentions = set()

    # 7. Przetwarzanie kolejnych okien

    null_count = 0
    antecedent_count = 0

    all_window_mentions = []

    for window_idx, (window_result, span) in enumerate(
            zip(document_outputs, spans)
    ):

        # Sprawdzenie struktury okna

        if not isinstance(window_result, tuple):
            raise RuntimeError(
                f"Okno {window_idx}: "
                f"oczekiwano tuple, otrzymano "
                f"{type(window_result)}"
            )

        if len(window_result) != 6:
            raise RuntimeError(
                f"Okno {window_idx}: "
                f"oczekiwano 6 elementów, "
                f"otrzymano {len(window_result)}"
            )

        (
            loss,
            mention_start_ids,
            mention_end_ids,
            final_logits,
            mention_logits,
            span_mask
        ) = window_result

        # Granice okna

        window_start = int(span[0])
        window_end = int(span[1])

        # Usunięcie wymiaru batch

        mention_start_ids = mention_start_ids[0]
        mention_end_ids = mention_end_ids[0]
        final_logits = final_logits[0]
        span_mask = span_mask[0]

        # Liczba mentionów

        num_mentions = mention_start_ids.shape[0]

        # Sprawdzenie wymiarów

        if mention_end_ids.shape[0] != num_mentions:
            raise RuntimeError(
                f"Okno {window_idx}: "
                "mention_start_ids i mention_end_ids "
                "mają różne rozmiary."
            )

        if final_logits.shape[0] != num_mentions:
            raise RuntimeError(
                f"Okno {window_idx}: "
                f"final_logits ma "
                f"{final_logits.shape[0]} mentionów, "
                f"oczekiwano {num_mentions}."
            )

        # 8. Aktywni kandydaci

        valid_indices = [
            i
            for i in range(num_mentions)
            if bool(span_mask[i].item())
        ]

        # 9. Mapowanie indeks kandydata -> globalny span

        mentions_by_index = {}

        for i in valid_indices:
            local_start = int(
                mention_start_ids[i].item()
            )

            local_end = int(
                mention_end_ids[i].item()
            )

            global_start = local_start + window_start
            global_end = local_end + window_start

            mentions_by_index[i] = (
                global_start,
                global_end
            )

        # 10. Najlepszy antecedent / NULL

        best_antecedents = torch.argmax(
            final_logits,
            dim=-1
        )

        max_k = num_mentions

        for i in valid_indices:

            antecedent_idx = int(
                best_antecedents[i].item()
            )

            mention = mentions_by_index[i]

            mention_windows[mention].add(window_idx)

            all_window_mentions.append(
                (mention, window_idx)
            )

            # NULL

            if antecedent_idx == max_k:
                null_count += 1
                continue

            antecedent_count += 1

            # Nieprawidłowy indeks

            if antecedent_idx not in mentions_by_index:
                continue

            antecedent = mentions_by_index[
                antecedent_idx
            ]

            # Samoodwołanie

            if mention == antecedent:
                continue

            # Mamy rzeczywistą relację koreferencyjną

            all_mentions.add(mention)
            all_mentions.add(antecedent)

            # mention_to_antecedent[
            #    mention
            # ] = antecedent

            mention_antecedent_votes[mention].append(
                {
                    "window": window_idx,
                    "antecedent": antecedent,
                    "score": float(
                        final_logits[i, antecedent_idx].item()
                    )
                }
            )

    # 11. Union-Find

    def build_clusters_from_antecedents(
            mention_to_antecedent
    ):
        """
        Buduje klastry koreferencji na podstawie
        relacji mention -> antecedent.
        """

        all_mentions = set()

        for mention, antecedent in mention_to_antecedent.items():
            all_mentions.add(mention)
            all_mentions.add(antecedent)

        parent = {
            mention: mention
            for mention in all_mentions
        }

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            root_x = find(x)
            root_y = find(y)

            if root_x != root_y:
                parent[root_y] = root_x

        for mention, antecedent in mention_to_antecedent.items():
            union(mention, antecedent)

        clusters_dict = defaultdict(list)

        for mention in all_mentions:
            root = find(mention)
            clusters_dict[root].append(mention)

        return list(clusters_dict.values())

    mention_to_antecedent = aggregate_antecedent_votes(
        mention_antecedent_votes
    )

    predicted_clusters = build_clusters_from_antecedents(mention_to_antecedent)

    # 15. Sortowanie

    for cluster in predicted_clusters:
        cluster.sort(
            key=lambda x: (x[0], x[1])
        )

    predicted_clusters.sort(
        key=lambda cluster: (
            cluster[0][0],
            cluster[0][1]
        )
    )

    # 16. Diagnostyka

    print()
    print(
        "Liczba wykrytych mentionów:",
        len(all_mentions)
    )

    print(
        "Liczba relacji mention -> antecedent:",
        len(mention_to_antecedent)
    )

    print(
        "Liczba przewidywanych klastrów:",
        len(predicted_clusters)
    )

    print(
        "Decyzje NULL:",
        null_count
    )

    print(
        "Decyzje z antecedentem:",
        antecedent_count
    )

    return predicted_clusters


def clusters_to_pairs(clusters):
    """
    Zamienia klastry koreferencji na pary mention -> antecedent.
    """

    pairs = set()

    for cluster in clusters:

        # Singleton nie tworzy relacji koreferencyjnej
        if len(cluster) < 2:
            continue

        cluster = sorted(
            cluster,
            key=lambda x: (x[0], x[1])
        )

        for i in range(1, len(cluster)):

            mention = cluster[i]

            for j in range(i):

                antecedent = cluster[j]

                if mention != antecedent:
                    pairs.add(
                        (mention, antecedent)
                    )

    return pairs


def span_to_text(span, input_ids, tokenizer):
    """
    Zamienia span tokenów (start, end) na tekst.
    """

    start, end = span

    token_ids = input_ids[start:end + 1]

    return tokenizer.decode(
        token_ids,
        skip_special_tokens=True
    ).strip()


def compare_base_finetuned_gold(
        base_clusters,
        finetuned_clusters,
        gold_clusters,
        input_ids,
        tokenizer
):
    """
    Porównuje Base, Fine-tuned oraz Gold.

    Zwraca słownik zawierający:

        NEW_CORRECT
            Fine-tuned poprawny,
            Base błędny/brak,
            Gold poprawny.

        LOST_CORRECT
            Base poprawny,
            Fine-tuned błędny/brak,
            Gold poprawny.

        CORRECT_BOTH
            Oba modele poprawne.

        NEW_ERROR
            Fine-tuned tworzy nową błędną relację.

        CORRECTED_ERROR
            Base miał błąd,
            Fine-tuned go usunął.

        WRONG_BOTH
            Oba modele mają tę samą błędną relację.
    """

    # ---------------------------------------------------------
    # 1. Zamiana klastrów na relacje
    # ---------------------------------------------------------

    base_pairs = clusters_to_pairs(
        base_clusters
    )

    finetuned_pairs = clusters_to_pairs(
        finetuned_clusters
    )

    gold_pairs = clusters_to_pairs(
        gold_clusters
    )

    # ---------------------------------------------------------
    # 2. Kategorie
    # ---------------------------------------------------------

    new_correct = (
                          finetuned_pairs
                          & gold_pairs
                  ) - base_pairs

    lost_correct = (
                           base_pairs
                           & gold_pairs
                   ) - finetuned_pairs

    correct_both = (
            base_pairs
            & finetuned_pairs
            & gold_pairs
    )

    new_errors = (
                         finetuned_pairs
                         - gold_pairs
                 ) - base_pairs

    corrected_errors = (
                               base_pairs
                               - gold_pairs
                       ) - finetuned_pairs

    wrong_both = (
                         base_pairs
                         & finetuned_pairs
                 ) - gold_pairs

    # ---------------------------------------------------------
    # 3. Wszystkie relacje
    # ---------------------------------------------------------

    all_pairs = (
            base_pairs
            | finetuned_pairs
            | gold_pairs
    )

    details = []

    # ---------------------------------------------------------
    # 4. Szczegółowe porównanie każdej relacji
    # ---------------------------------------------------------

    for mention, antecedent in sorted(
            all_pairs,
            key=lambda x: (
                    x[0][0],
                    x[0][1],
                    x[1][0],
                    x[1][1]
            )
    ):

        pair = (
            mention,
            antecedent
        )

        in_base = pair in base_pairs
        in_finetuned = pair in finetuned_pairs
        in_gold = pair in gold_pairs

        # -----------------------------------------------------
        # Klasyfikacja
        # -----------------------------------------------------

        if in_gold and not in_base and in_finetuned:
            category = "NEW_CORRECT"

        elif in_gold and in_base and not in_finetuned:
            category = "LOST_CORRECT"

        elif in_gold and in_base and in_finetuned:
            category = "CORRECT_BOTH"

        elif not in_gold and not in_base and in_finetuned:
            category = "NEW_ERROR"

        elif not in_gold and in_base and not in_finetuned:
            category = "CORRECTED_ERROR"

        elif not in_gold and in_base and in_finetuned:
            category = "WRONG_BOTH"

        else:
            category = "OTHER"

        # -----------------------------------------------------
        # Tekst
        # -----------------------------------------------------

        mention_text = span_to_text(
            mention,
            input_ids,
            tokenizer
        )

        antecedent_text = span_to_text(
            antecedent,
            input_ids,
            tokenizer
        )

        details.append(
            {
                "mention": mention,
                "mention_text": mention_text,

                "antecedent": antecedent,
                "antecedent_text": antecedent_text,

                "base": in_base,
                "finetuned": in_finetuned,
                "gold": in_gold,

                "category": category
            }
        )

    # ---------------------------------------------------------
    # 5. Podsumowanie
    # ---------------------------------------------------------

    summary = {
        "base_relations": len(base_pairs),
        "finetuned_relations": len(finetuned_pairs),
        "gold_relations": len(gold_pairs),

        "new_correct": len(new_correct),
        "lost_correct": len(lost_correct),
        "correct_both": len(correct_both),

        "new_errors": len(new_errors),
        "corrected_errors": len(corrected_errors),
        "wrong_both": len(wrong_both)
    }

    return {
        "summary": summary,

        "new_correct": new_correct,
        "lost_correct": lost_correct,
        "correct_both": correct_both,

        "new_errors": new_errors,
        "corrected_errors": corrected_errors,
        "wrong_both": wrong_both,

        "details": details
    }


def print_comparison_report(comparison):
    """
    Wyświetla podsumowanie porównania
    Base / Fine-tuned / Gold.
    """

    summary = comparison["summary"]

    print()
    print("=" * 80)
    print("PORÓWNANIE BASE / FINE-TUNED / GOLD")
    print("=" * 80)

    print()
    print("Liczba relacji:")
    print(
        f"  Base:       {summary['base_relations']}"
    )
    print(
        f"  Fine-tuned: {summary['finetuned_relations']}"
    )
    print(
        f"  Gold:       {summary['gold_relations']}"
    )

    print()
    print("Zmiany po fine-tuningu:")
    print(
        f"  Nowe poprawne:       {summary['new_correct']}"
    )
    print(
        f"  Utracone poprawne:   {summary['lost_correct']}"
    )
    print(
        f"  Poprawne w obu:      {summary['correct_both']}"
    )
    print(
        f"  Nowe błędy:          {summary['new_errors']}"
    )
    print(
        f"  Poprawione błędy:    {summary['corrected_errors']}"
    )
    print(
        f"  Błędne w obu:        {summary['wrong_both']}"
    )


# ============================================================
# METRYKI
# ============================================================

def run_reference_scorer(scorer_path, metric, gold_file, predicted_file):
    import subprocess
    import os

    perl_exe = r"C:\Strawberry\perl\bin\perl.exe"

    scorer_path = os.path.abspath(scorer_path)

    scorer_dir = os.path.dirname(scorer_path)

    lib_dir = os.path.join(
        scorer_dir,
        "lib"
    )

    result = subprocess.run(
        [
            perl_exe,
            f"-I{lib_dir}",
            scorer_path,
            metric,
            gold_file,
            predicted_file,
            "none"
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace"
    )

    if result.returncode != 0:
        raise RuntimeError(
            "Reference Scorer zakończył się błędem.\n"
            f"Kod: {result.returncode}\n\n"
            f"STDOUT:\n{result.stdout}\n\n"
            f"STDERR:\n{result.stderr}"
        )

    return result.stdout


def parse_coreference_scores(output, metric_name):
    import re

    identification_match = re.search(
        r"Identification of Mentions:\s*"
        r"Recall:\s*\([^)]*\)\s*([0-9]+(?:\.[0-9]+)?)%\s*"
        r"Precision:\s*\([^)]*\)\s*([0-9]+(?:\.[0-9]+)?)%\s*"
        r"F1:\s*([0-9]+(?:\.[0-9]+)?)%",
        output,
        re.IGNORECASE
    )

    identification = None

    if identification_match:
        identification = {
            "recall": float(
                identification_match.group(1)
            ),
            "precision": float(
                identification_match.group(2)
            ),
            "f1": float(
                identification_match.group(3)
            )
        }

    coref_match = re.search(
        r"Coreference:\s*"
        r"Recall:\s*\([^)]*\)\s*([0-9]+(?:\.[0-9]+)?)%\s*"
        r"Precision:\s*\([^)]*\)\s*([0-9]+(?:\.[0-9]+)?)%\s*"
        r"F1:\s*([0-9]+(?:\.[0-9]+)?)%",
        output,
        re.IGNORECASE
    )

    if not coref_match:
        raise RuntimeError(
            f"Nie udało się odczytać wyników {metric_name}."
        )

    return {
        "precision": float(
            coref_match.group(2)
        ),
        "recall": float(
            coref_match.group(1)
        ),
        "f1": float(
            coref_match.group(3)
        ),
        "identification": identification
    }


def evaluate_with_reference_scorer(scorer_path, gold_file, predicted_file):
    """
    Oblicza MUC, B³, CEAF-e oraz CoNLL F1
    za pomocą Reference Coreference Scorer.

    Reference Scorer jest uruchamiany osobno dla:
        - muc
        - bcub
        - ceafe
    """

    print()
    print("=" * 70)
    print("REFERENCE SCORER")
    print("=" * 70)

    # ---------------------------------------------------------
    # Uruchomienie scorera dla poszczególnych metryk
    # ---------------------------------------------------------

    metrics = {
        "muc": "MUC",
        "bcub": "B³",
        "ceafe": "CEAF-e"
    }

    results = {}

    identification_scores = None

    for metric, metric_name in metrics.items():
        print()
        print(f"Obliczanie {metric_name}...")

        output = run_reference_scorer(
            scorer_path=scorer_path,
            metric=metric,
            gold_file=gold_file,
            predicted_file=predicted_file
        )

        print(f"\n--- OUTPUT {metric_name} ---")
        print(output)

        scores = parse_coreference_scores(
            output,
            metric_name
        )

        results[metric] = {
            "precision": scores["precision"],
            "recall": scores["recall"],
            "f1": scores["f1"]
        }

        if identification_scores is None:
            identification_scores = scores["identification"]

    # ---------------------------------------------------------
    # CoNLL F1
    # ---------------------------------------------------------

    results["identification"] = identification_scores

    conll_f1 = (
                       results["muc"]["f1"]
                       + results["bcub"]["f1"]
                       + results["ceafe"]["f1"]
               ) / 3.0

    results["conll_f1"] = conll_f1

    print()
    print("=" * 70)
    print("WYNIKI REFERENCE SCORER")
    print("=" * 70)

    print(
        f"Identification of Mentions: "
        f"P={results['identification']['precision']:.2f}%  "
        f"R={results['identification']['recall']:.2f}%  "
        f"F1={results['identification']['f1']:.2f}%"
    )

    print("-" * 70)

    print(
        f"MUC:     "
        f"P={results['muc']['precision']:.2f}%  "
        f"R={results['muc']['recall']:.2f}%  "
        f"F1={results['muc']['f1']:.2f}%"
    )

    print(
        f"B³:      "
        f"P={results['bcub']['precision']:.2f}%  "
        f"R={results['bcub']['recall']:.2f}%  "
        f"F1={results['bcub']['f1']:.2f}%"
    )

    print(
        f"CEAF-e:  "
        f"P={results['ceafe']['precision']:.2f}%  "
        f"R={results['ceafe']['recall']:.2f}%  "
        f"F1={results['ceafe']['f1']:.2f}%"
    )

    print("-" * 70)

    print(
        f"CoNLL F1: {results['conll_f1']:.2f}%"
    )

    print("=" * 70)

    return results


def safe_divide(
        numerator,
        denominator
):
    if denominator == 0:
        return 0.0

    return numerator / denominator


# ============================================================
# RAPORT
# ============================================================


def get_span_context(
        span1,
        span2,
        input_ids,
        tokenizer,
        context_tokens=20
):
    """
    Zwraca tekst dwóch mentionów oraz ich lokalny kontekst.

    Dla każdej wzmianki pobierane jest:
        - context_tokens tokenów przed wzmianką,
        - sama wzmianka,
        - context_tokens tokenów po wzmiance.

    Wzmianki są oznaczane w kontekście za pomocą:
        <<< MENTION >>>

    span1, span2:
        (start, end)
    """

    start1, end1 = span1
    start2, end2 = span2

    if start2 < start1:
        span1, span2 = span2, span1
        start1, end1 = span1
        start2, end2 = span2

    sequence_length = input_ids.shape[1]

    # ---------------------------------------------------------
    # Funkcja pomocnicza tworząca lokalny kontekst
    # ---------------------------------------------------------

    def build_context(start, end):
        context_start = max(
            0,
            start - context_tokens
        )

        context_end = min(
            sequence_length,
            end + context_tokens + 1
        )

        # Tokeny przed mentionem
        before_ids = input_ids[
                     0,
                     context_start:start
                     ].tolist()

        # Tokeny mentionu
        mention_ids = input_ids[
                      0,
                      start:end + 1
                      ].tolist()

        # Tokeny po mentionie
        after_ids = input_ids[
                    0,
                    end + 1:context_end
                    ].tolist()

        before_text = tokenizer.decode(
            before_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        ).strip()

        mention_text = tokenizer.decode(
            mention_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        ).strip()

        after_text = tokenizer.decode(
            after_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        ).strip()

        # -----------------------------------------------------
        # Kontekst z oznaczoną wzmianką
        # -----------------------------------------------------

        context_text = (
            f"{before_text} "
            f"<<< {mention_text} >>> "
            f"{after_text}"
        ).strip()

        return mention_text, context_text

    # ---------------------------------------------------------
    # Kontekst obu wzmianek
    # ---------------------------------------------------------

    mention1_text, context1 = build_context(
        start1,
        end1
    )

    mention2_text, context2 = build_context(
        start2,
        end2
    )

    # ---------------------------------------------------------
    # Wynik
    # ---------------------------------------------------------

    return {
        "mention1": mention1_text,
        "mention2": mention2_text,

        "context1": context1,
        "context2": context2,

        "span1": span1,
        "span2": span2
    }


def extract_cases(
        category,
        base_clusters,
        finetuned_clusters,
        gold_clusters,
        input_ids,
        tokenizer,
        context_tokens=20
):
    """
    Wyszukuje przypadki należące do wskazanej kategorii.

    Obsługiwane kategorie:

        NEW_CORRECT
            Relacja:
                - występuje w Fine-tuned,
                - występuje w Gold,
                - nie występuje w Base.

        LOST_CORRECT
            Relacja:
                - występuje w Base,
                - występuje w Gold,
                - nie występuje w Fine-tuned.

        WRONG_BOTH
            Relacja:
                - występuje w Base,
                - występuje w Fine-tuned,
                - nie występuje w Gold.

    Zwraca listę przypadków wraz z tekstem mentionów,
    spanami oraz kontekstem.
    """

    base_pairs = clusters_to_pairs(base_clusters)
    finetuned_pairs = clusters_to_pairs(finetuned_clusters)
    gold_pairs = clusters_to_pairs(gold_clusters)

    # ---------------------------------------------------------
    # Wyznaczenie relacji dla poszczególnych kategorii
    # ---------------------------------------------------------

    if category == "NEW_CORRECT":

        cases = (
                finetuned_pairs
                & gold_pairs
                - base_pairs
        )

        title = "NOWE POPRAWNE RELACJE"

    elif category == "LOST_CORRECT":

        cases = (
                base_pairs
                & gold_pairs
                - finetuned_pairs
        )

        title = "UTRACONE POPRAWNE RELACJE"

    elif category == "WRONG_BOTH":

        cases = (
                base_pairs
                & finetuned_pairs
                - gold_pairs
        )

        title = "BŁĘDNE RELACJE W OBU MODELACH"
    elif category == "MISSING_BOTH":
        cases = (
                gold_pairs
                - finetuned_pairs
                - base_pairs
        )

        title = "BRAK RELACJI W OBU MODELACH"
    else:
        raise ValueError(
            f"Nieznana kategoria: {category}. "
            f"Dozwolone: NEW_CORRECT, LOST_CORRECT, WRONG_BOTH."
        )

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)
    print("Liczba:", len(cases))

    # ---------------------------------------------------------
    # Przygotowanie wyników
    # ---------------------------------------------------------

    results = []

    for pair in sorted(
            cases,
            key=lambda p: (
                    p[0][0],
                    p[0][1],
                    p[1][0],
                    p[1][1]
            )
    ):
        span1, span2 = pair

        data = get_span_context(
            span1,
            span2,
            input_ids,
            tokenizer,
            context_tokens=context_tokens
        )

        data["category"] = category

        results.append(data)

    return results


def save_cases_to_csv(
        cases,
        filename
):
    """
    Zapisuje przypadki do pliku CSV.
    """

    with open(
            filename,
            "w",
            newline="",
            encoding="utf-8-sig"
    ) as f:
        writer = csv.writer(f)

        writer.writerow([
            "ID",
            "Mention 1",
            "Mention 2",
            "Span 1",
            "Span 2",
            "Context 1",
            "Context 2"
        ])

        for i, case in enumerate(cases, start=1):
            writer.writerow([
                i,
                case["mention1"],
                case["mention2"],
                str(case["span1"]),
                str(case["span2"]),
                case["context1"],
                case["context2"]
            ])

    print(
        f"Zapisano {len(cases)} przypadków do: "
        f"{filename}"
    )


# ============================================================
# MAIN
# ============================================================

def main():
    print()
    print("=" * 80)
    print("HERFERENCE - TEST SET EVALUATION")
    print("=" * 80)

    print(
        "DEVICE:",
        DEVICE
    )

    print(
        "BASE MODEL:",
        MODEL_NAME
    )

    print(
        "FINE-TUNED:",
        FINETUNED_MODEL
    )

    # ========================================================
    # DOKUMENT
    # ========================================================

    document = load_test_document()

    # ========================================================
    # GOLD
    # ========================================================

    print()
    print(
        "Przygotowywanie gold clusters..."
    )

    # ========================================================
    # MODEL BAZOWY
    # ========================================================

    base_model, base_tokenizer = (
        load_model(
            MODEL_NAME
        )
    )

    (
        words,
        input_ids,
        attention_mask,
        word_to_tokens
    ) = tokenize_document(
        document,
        base_tokenizer
    )

    gold_clusters = (
        convert_gold_clusters(
            document,
            word_to_tokens
        )
    )
    singleton_gold = sum(
        1 for cluster in gold_clusters
        if len(cluster) == 1
    )

    multi_gold = sum(
        1 for cluster in gold_clusters
        if len(cluster) > 1
    )

    print()
    print("Gold clusters:", len(gold_clusters))
    print("Gold singleton clusters:", singleton_gold)
    print("Gold multi-mention clusters:", multi_gold)

    print()
    print(
        "Gold clusters:",
        len(gold_clusters)
    )

    input_ids = input_ids.unsqueeze(0).to(
        DEVICE
    )

    attention_mask = (
        attention_mask
        .unsqueeze(0)
        .to(DEVICE)
    )

    print()
    print("=" * 80)
    print("MODEL BAZOWY")
    print("=" * 80)

    base_predictions = predict(
        base_model,
        input_ids,
        attention_mask,
        DEVICE,
        gold_clusters=gold_clusters
    )

    gold_mentions = set(
        mention
        for cluster in gold_clusters
        for mention in cluster
    )

    predicted_mentions = set(
        mention
        for cluster in base_predictions
        for mention in cluster
    )

    print()
    print("=== DETEKCJA MENTIONÓW — BASE ===")

    print(
        "Gold mentions:",
        len(gold_mentions)
    )

    print(
        "Predicted mentions:",
        len(predicted_mentions)
    )

    print()
    print("=== DIAGNOSTYKA BASE ===")

    print(
        "Liczba gold clusters:",
        len(gold_clusters)
    )

    print(
        "Liczba predicted clusters:",
        len(base_predictions)
    )

    gold_mentions = sum(
        len(cluster)
        for cluster in gold_clusters
    )

    gold_mention_to_cluster = {}

    for cluster_id, cluster in enumerate(gold_clusters):
        for mention in cluster:
            gold_mention_to_cluster[mention] = cluster_id

    pred_mentions = sum(
        len(cluster)
        for cluster in base_predictions
    )

    print(
        "Liczba gold mentions:",
        gold_mentions
    )

    print(
        "Liczba predicted mentions:",
        pred_mentions
    )

    # --------------------------------------------------------
    # Zwolnienie GPU
    # --------------------------------------------------------

    del base_model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ========================================================
    # MODEL FINE-TUNED
    # ========================================================

    fine_model, fine_tokenizer = (
        load_model(
            FINETUNED_MODEL
        )
    )

    (
        words,
        input_ids,
        attention_mask,
        word_to_tokens
    ) = tokenize_document(
        document,
        fine_tokenizer
    )

    gold_clusters_fine = (
        convert_gold_clusters(
            document,
            word_to_tokens
        )
    )

    input_ids = input_ids.unsqueeze(0).to(
        DEVICE
    )

    attention_mask = (
        attention_mask
        .unsqueeze(0)
        .to(DEVICE)
    )

    print()
    print("=" * 80)
    print("MODEL PO FINE-TUNINGU")
    print("=" * 80)

    fine_predictions = predict(
        fine_model,
        input_ids,
        attention_mask,
        DEVICE,
        gold_clusters=gold_clusters
    )

    gold_mentions = set(
        mention
        for cluster in gold_clusters
        for mention in cluster
    )

    predicted_mentions = set(
        mention
        for cluster in fine_predictions
        for mention in cluster
    )

    # ========================================================
    # ZAPIS DO JSON
    # ========================================================

    tokens = input_ids_to_tokens(
        input_ids,
        base_tokenizer
    )

    write_conll_file(
        filename="gold.conll",
        tokens=tokens,
        clusters=gold_clusters,
        document_id="test"
    )

    write_conll_file(
        filename="base.conll",
        tokens=tokens,
        clusters=base_predictions,
        document_id="test"
    )

    write_conll_file(
        filename="finetuned.conll",
        tokens=tokens,
        clusters=fine_predictions,
        document_id="test"
    )

    SCORER = r"C:\Users\Tomek\Desktop\magisterka\magisterka\reference-coreference-scorers\scorer.pl"

    base_results = evaluate_with_reference_scorer(scorer_path=SCORER,
                                                  gold_file="C:\\Users\Tomek\Desktop\magisterka\magisterka\gold.conll",
                                                  predicted_file="C:\\Users\Tomek\Desktop\magisterka\magisterka\\base.conll")

    fine_results = evaluate_with_reference_scorer(scorer_path=SCORER,
                                                  gold_file="C:\\Users\Tomek\Desktop\magisterka\magisterka\gold.conll",
                                                  predicted_file="C:\\Users\Tomek\Desktop\magisterka\magisterka\\finetuned.conll")

    results = {
        "test_document": document.get(
            "doc_key",
            "unknown"
        ),

        "base_model": {
            "model": MODEL_NAME,
            "metrics": base_results
        },

        "fine_tuned_model": {
            "model": FINETUNED_MODEL,
            "metrics": fine_results
        }
    }

    comparison = compare_base_finetuned_gold(
        base_clusters=base_predictions,
        finetuned_clusters=fine_predictions,
        gold_clusters=gold_clusters,
        input_ids=input_ids,
        tokenizer=base_tokenizer
    )

    print_comparison_report(
        comparison
    )

    new_correct_cases = extract_cases(category="NEW_CORRECT",
                                      base_clusters=base_predictions,
                                      finetuned_clusters=fine_predictions,
                                      gold_clusters=gold_clusters,
                                      input_ids=input_ids,
                                      tokenizer=base_tokenizer,
                                      context_tokens=20
                                      )

    lost_correct_cases = extract_cases(category="LOST_CORRECT",
                                       base_clusters=base_predictions,
                                       finetuned_clusters=fine_predictions,
                                       gold_clusters=gold_clusters,
                                       input_ids=input_ids,
                                       tokenizer=base_tokenizer,
                                       context_tokens=20
                                       )

    wrong_both_cases = extract_cases(category="WRONG_BOTH",
                                     base_clusters=base_predictions,
                                     finetuned_clusters=fine_predictions,
                                     gold_clusters=gold_clusters,
                                     input_ids=input_ids,
                                     tokenizer=base_tokenizer,
                                     context_tokens=20
                                     )

    missing_both_cases = extract_cases(category="MISSING_BOTH",
                                       base_clusters=base_predictions,
                                       finetuned_clusters=fine_predictions,
                                       gold_clusters=gold_clusters,
                                       input_ids=input_ids,
                                       tokenizer=base_tokenizer,
                                       context_tokens=20
                                       )

    nc_path = os.path.join(
        "cases",
        EXPERIMENT_NAME + VERSION,
        "new_correct_cases.csv"
    )

    lc_path = os.path.join(
        "cases",
        EXPERIMENT_NAME + VERSION,
        "lost_correct_cases.csv"
    )

    mb_path = os.path.join(
        "cases",
        EXPERIMENT_NAME + VERSION,
        "missing_both_cases.csv"
    )

    wb_path = os.path.join(
        "cases",
        EXPERIMENT_NAME + VERSION,
        "wrong_both_cases.csv"
    )

    save_cases_to_csv(
        new_correct_cases,
        nc_path
    )

    save_cases_to_csv(lost_correct_cases, lc_path)

    save_cases_to_csv(wrong_both_cases, wb_path)

    save_cases_to_csv(missing_both_cases, mb_path)

    output_file = os.path.join(
        "cases",
        EXPERIMENT_NAME + VERSION,
        "test_evaluation_results.csv"
    )

    with open(
            output_file,
            "w",
            encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            ensure_ascii=False,
            indent=4
        )

    print()
    print(
        "Wyniki zapisane do:",
        output_file
    )

    print()
    print("=" * 80)
    print("EWALUACJA ZAKOŃCZONA")
    print("=" * 80)


if __name__ == "__main__":
    main()
