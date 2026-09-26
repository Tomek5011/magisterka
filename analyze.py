import re
from collections import defaultdict

def analyze_webanno(input_path):
    mentions = defaultdict(list)
    clusters = defaultdict(set)

    total_annotation_occurrences = 0
    lines_with_coref = 0

    print("\n\n========================================")
    print("PLIK:", input_path)
    print("========================================")

    with open(input_path, "r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, 1):

            line = raw_line.rstrip("\r\n")

            if not line or line.startswith("#"):
                continue

            parts = line.split("\t")

            if len(parts) < 4:
                continue

            token_id = parts[0]
            token_text = parts[2]

            # Wszystkie kolumny po tekście tokena
            coref_info = "\t".join(parts[3:])

            # Wszystkie linki do poprzednich elementów łańcucha
            link_ids = re.findall(
                r"\*->(\d+-\d+)",
                coref_info
            )

            cluster_ids = re.findall(
                r"(?:\*|\d+)\[(\d+)\]",
                coref_info
            )

            if not link_ids and not cluster_ids:
                continue

            lines_with_coref += 1

            # Diagnostyka
            if link_ids and not cluster_ids:
                print(
                    f"\n!!! BRAK KLASTRA w linii {line_no}"
                )
                print(raw_line.rstrip())
                print("link_ids:", link_ids)
                print("cluster_ids:", cluster_ids)

            # Jeżeli liczba linków i klastrów się różni,
            # pokaż ostrzeżenie, ale nie pomijaj całej adnotacji.
            if len(link_ids) != len(cluster_ids):
                print(
                    f"\n!!! NIEZGODNOŚĆ w linii {line_no}"
                )
                print(raw_line.rstrip())
                print("link_ids:", link_ids)
                print("cluster_ids:", cluster_ids)

            # --------------------------------------------------
            # Zbieranie informacji o wzmiankach
            # --------------------------------------------------

            for link_id in link_ids:

                total_annotation_occurrences += 1

                mentions[link_id].append(
                    (token_id, token_text, line_no)
                )

            # --------------------------------------------------
            # Przypisanie linków do klastrów
            # --------------------------------------------------

            if cluster_ids:

                # W typowym WebAnno TSV informacja o klastrze
                # dotyczy wszystkich linków znajdujących się
                # w danej adnotacji.
                for cluster_id in cluster_ids:

                    for link_id in link_ids:
                        clusters[cluster_id].add(link_id)

    # ----------------------------------------------------------
    # Statystyki
    # ----------------------------------------------------------

    print("\n==============================")
    print("DIAGNOSTYKA")
    print("==============================")

    print(
        "Liczba wystąpień adnotacji na tokenach:",
        total_annotation_occurrences
    )

    print(
        "Liczba unikalnych wzmianek:",
        len(mentions)
    )

    print(
        "Liczba klastrów:",
        len(clusters)
    )

    print(
        "Liczba linii zawierających koreferencję:",
        lines_with_coref
    )


# --------------------------------------------------------------
# Analiza dokumentów
# --------------------------------------------------------------

analyze_webanno(
    "tsv_files/DU_2025_1661-uklady-zbiorowe-pracy.docx.tsv"
)

# analyze_webanno(
#     "tsv_files/DU_2026_465-dzialalnosc-kosmiczna.docx.tsv"
# )
#
# analyze_webanno(
#     "tsv_files/DU_2025_1017-krajowy-system-certyfikacji.docx.tsv"
# )
#
# analyze_webanno(
#     "tsv_files/DU_2025_1080-szczegolne-zasady-przygotowania.docx.tsv"
# )
#
# analyze_webanno(
#     "tsv_files/DU_2025_779-krajowa-siec-kardiologiczna.docx.tsv"
# )
#
# analyze_webanno(
#     "tsv_files/DU_2025_1826-nadzor-nad-ogolnym.docx.tsv"
# )
#
# analyze_webanno(
#     "tsv_files/DU_2025_1235-certyfikacja-wykonawcow.docx.tsv"
# )
#
# analyze_webanno(
#     "tsv_files/Ustawa_o_ochronie_sygnalistow.docx.tsv"
# )
