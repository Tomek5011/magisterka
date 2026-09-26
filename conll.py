import os
import re
import subprocess
from collections import defaultdict


# ============================================================
# CONLL FORMAT
# ============================================================

def normalize_clusters(clusters):
    """
    Normalizuje klastry do postaci:

        [
            [(start, end), (start, end)],
            ...
        ]

    gdzie start/end są indeksami tokenów w całym dokumencie.
    """

    normalized = []

    for cluster in clusters:

        clean_cluster = []

        for mention in cluster:

            if not isinstance(mention, (tuple, list)):
                raise ValueError(
                    f"Nieprawidłowy mention: {mention!r}"
                )

            if len(mention) != 2:
                raise ValueError(
                    f"Mention powinien mieć 2 elementy: "
                    f"{mention!r}"
                )

            start = int(mention[0])
            end = int(mention[1])

            if start < 0 or end < start:
                raise ValueError(
                    f"Nieprawidłowy span: {(start, end)}"
                )

            clean_cluster.append(
                (start, end)
            )

        if clean_cluster:
            normalized.append(clean_cluster)

    return normalized


# ============================================================
# MAPA TOKEN -> KLASTRY
# ============================================================

def clusters_to_conll_annotations(clusters):
    """
    Zamienia klastry:

        [
            [(1, 3), (8, 9)],
            [(15, 15)]
        ]

    na adnotacje tokenowe CoNLL.

    Zwraca:

        annotations[token_index] = "(1"
        annotations[token_index] = "1)"
        itd.

    Obsługuje również:
        - pojedynczy mention,
        - wiele mentionów zaczynających się na tym samym tokenie,
        - zagnieżdżone mentiony.

    Format:
        (cluster_id
        cluster_id)
        (cluster_id)
        (cluster_id
    """

    clusters = normalize_clusters(clusters)

    annotations = defaultdict(list)

    for cluster_id, cluster in enumerate(
        clusters,
        start=1
    ):

        for start, end in cluster:

            if start == end:

                annotations[start].append(
                    f"({cluster_id})"
                )

            else:

                annotations[start].append(
                    f"({cluster_id}"
                )

                annotations[end].append(
                    f"{cluster_id})"
                )

    return annotations


# ============================================================
# CONLL FILE
# ============================================================

def write_conll_file(
    filename,
    tokens,
    clusters,
    document_id="document"
):
    """
    Tworzy plik CoNLL-2012.

    tokens:
        lista tokenów dokumentu.

    clusters:
        klastry koreferencji w indeksach tokenów.
    """

    annotations = clusters_to_conll_annotations(
        clusters
    )

    with open(
        filename,
        "w",
        encoding="utf-8"
    ) as f:

        # ----------------------------------------------------
        # Początek dokumentu
        # ----------------------------------------------------

        f.write(
            f"#begin document ({document_id}); part 000\n"
        )

        # ----------------------------------------------------
        # Tokeny
        # ----------------------------------------------------

        for token_idx, token in enumerate(tokens):

            coref = annotations.get(
                token_idx,
                []
            )

            if coref:

                coref_string = "|".join(
                    coref
                )

            else:

                coref_string = "-"

            # Minimalny format:
            #
            # doc_id
            # part
            # token_id
            # token
            # coreference
            #
            f.write(
                f"{document_id}\t"
                f"000\t"
                f"{token_idx}\t"
                f"{token}\t"
                f"{coref_string}\n"
            )

        # ----------------------------------------------------
        # Koniec dokumentu
        # ----------------------------------------------------

        f.write(
            "#end document\n"
        )


# ============================================================
# TOKENY Z INPUT_IDS
# ============================================================

def input_ids_to_tokens(
    input_ids,
    tokenizer
):
    """
    Zamienia input_ids na tokeny używane w pliku CoNLL.

    Ważne:
    tokeny muszą odpowiadać dokładnie tym pozycjom,
    których używają spans w clusters.
    """

    ids = input_ids[0].tolist()

    tokens = tokenizer.convert_ids_to_tokens(
        ids
    )

    return tokens