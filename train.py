import json
import torch
import os
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoConfig, AutoTokenizer
from herference.model import S2E
import numpy as np

# ====================== KONFIGURACJA ======================
MODEL_NAME = "ipipan/herference-large"
DATA_PATH = "train.jsonl"
BATCH_SIZE = 1
LEARNING_RATE = 2e-5
EPOCHS = 5
MAX_MODEL_INPUT = 512
MAX_SPAN_LENGTH = 30

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ModelParams:
    max_span_length = MAX_SPAN_LENGTH
    top_lambda = 0.4
    ffnn_size = 3072
    normalise_loss = True
    max_model_input_length = MAX_MODEL_INPUT
    dropout_prob = 0.3
    null_id_for_coref = 0

# ====================== DATASET ======================
class CorefDataset(Dataset):
    def __init__(self, path, tokenizer, max_len=1024):
        self.tokenizer = tokenizer
        self.max_len = max_len
        with open(path, 'r', encoding='utf-8') as f:
            self.data = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        words = [token for sentence in item['sentences'] for token in sentence]

        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            return_tensors="pt",
            #truncation=True,
            #max_length=self.max_len,
            padding="max_length"
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        word_ids = encoding.word_ids(batch_index=0)   # mapowanie token_id -> word_id

        # konwersja clusters z word-level na token-level
        token_clusters = []
        for cluster in item.get("clusters", []):
            token_cluster = []
            for word_start, word_end in cluster:
                token_start = None
                token_end = None
                for t_idx, w_idx in enumerate(word_ids):
                    if w_idx == word_start and token_start is None:
                        token_start = t_idx
                    if w_idx == word_end:
                        token_end = t_idx
                if token_start is not None and token_end is not None:
                    token_cluster.append([token_start, token_end])
            if token_cluster:
                token_clusters.append(token_cluster)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_clusters": token_clusters,   # lista list list [start_token, end_token]
        }


def collate_fn(batch):
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])

    # padding gold_clusters do kształtu [batch, max_num_clusters, max_mentions, 2]
    token_clusters_list = [item["token_clusters"] for item in batch]

    if not any(token_clusters_list):
        # wszystkie dokumenty puste
        padded = torch.zeros((len(batch), 1, 1, 2), dtype=torch.long, device=DEVICE)
    else:
        max_clusters = max(len(c) for c in token_clusters_list) if token_clusters_list else 1
        max_mentions = max((len(cl) for c in token_clusters_list for cl in c), default=1)

        padded = torch.full((len(batch), max_clusters, max_mentions, 2), 0, dtype=torch.long)

        for b, doc_clusters in enumerate(token_clusters_list):
            for c, cluster in enumerate(doc_clusters):
                for m, (start, end) in enumerate(cluster):
                    if m < max_mentions:
                        padded[b, c, m, 0] = start
                        padded[b, c, m, 1] = end

    return {
        "input_ids": input_ids.to(DEVICE),
        "attention_mask": attention_mask.to(DEVICE),
        "gold_clusters": padded.to(DEVICE)
    }


# ====================== TRENING ======================
def run_training():
    config = AutoConfig.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    params = ModelParams()

    model = S2E.from_pretrained(MODEL_NAME, config=config, params=params).to(DEVICE)

    dataset = CorefDataset(DATA_PATH, tokenizer, max_len=MAX_MODEL_INPUT)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        collate_fn=collate_fn, num_workers=0)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": 0.01},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=LEARNING_RATE)

    model.train()
    print(f"Start treningu na {DEVICE}")

    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        for step, batch in enumerate(loader):
            optimizer.zero_grad()

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                gold_clusters=batch["gold_clusters"]
            )

            loss = outputs[0]   # pierwszy element to loss gdy gold_clusters jest podany

            if torch.isnan(loss):
                print(f"NaN w lossie (step {step})")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()

            if step % 10 == 0:
                print(f"Epoch {epoch+1} | Step {step} | Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / len(loader)
        print(f"--- Epoch {epoch+1}/{EPOCHS} zakończony | Średnia loss: {avg_loss:.4f} ---\n")

    output_dir = "./finetuned_herference"
    os.makedirs(output_dir, exist_ok=True)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
    torch.save(vars(params), os.path.join(output_dir, "model_params.pt"))

    print(f"Model zapisany do: {output_dir}")


if __name__ == "__main__":
    run_training()