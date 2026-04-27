import json
import re
import os


def convert_webanno(input_path, output_path):
    sentences = []
    current_sentence = []

    # przechowuje wzmianki: {link_id: {"tokens": [indices], "cluster": id}}
    mentions_builder = {}
    global_token_idx = 0

    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()

        if not line or line.startswith("#"):
            if line.startswith("#Text=") and current_sentence:
                sentences.append(current_sentence)
                current_sentence = []
            continue

        parts = line.split()
        if len(parts) < 4:
            continue

        token_text = parts[2]
        current_sentence.append(token_text)

        # cała reszta linii to dane koreferencyjne
        coref_info = " ".join(parts[3:])

        # cluster id
        cluster_match = re.search(r'\[(\d+)\]', coref_info)

        # link id
        link_match = re.search(r'(\d+-\d+)', coref_info)

        if cluster_match and link_match:
            link_id = link_match.group(1)
            cluster_id = cluster_match.group(1)

            # unikalny klucz dla konkretnej wzmianki (np. "2-1" w klastrze "2")
            key = f"{link_id}_{cluster_id}"

            if key not in mentions_builder:
                mentions_builder[key] = {"tokens": [], "cluster": cluster_id}
            mentions_builder[key]["tokens"].append(global_token_idx)

        global_token_idx += 1

    if current_sentence:
        sentences.append(current_sentence)

    # budowanie klastrów z zebranych wzmianek
    clusters_dict = {}

    for key, data in mentions_builder.items():
        tokens = data["tokens"]
        if not tokens: continue

        span = [min(tokens), max(tokens)]
        c_id = data["cluster"]

        if c_id not in clusters_dict:
            clusters_dict[c_id] = []
        clusters_dict[c_id].append(span)

    # bierzemy tylko klastry z min. 2 wzmiankami
    final_clusters = [spans for spans in clusters_dict.values() if len(spans) > 1]

    output_data = {
        "doc_key": os.path.basename(input_path),
        "sentences": sentences,
        "clusters": final_clusters
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(output_data, ensure_ascii=False) + "\n")

    print(f"Liczba klastrów: {len(final_clusters)}")


convert_webanno("tsv_files/test.tsv", "train.jsonl")