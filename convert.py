import json
import re
import os
from collections import defaultdict

def validate_jsonl_data(data):
    """
    Kontrola poprawności danych przed zapisaniem JSONL.

    Sprawdza:
    1. obecność wymaganych pól,
    2. poprawność sentences,
    3. poprawność spanów,
    4. zakres indeksów tokenów,
    5. duplikaty spanów w klastrach,
    6. duplikaty klastrów,
    7. nakładanie się spanów w obrębie klastra,
    8. obecność singletonów,
    9. zgodność liczby tokenów z maksymalnym indeksem spanu.
    """

    print()
    print("=" * 80)
    print("KONTROLA JSONL")
    print("=" * 80)

    errors = []
    warnings = []

    # ------------------------------------------------------------------
    # 1. Wymagane pola
    # ------------------------------------------------------------------

    required_fields = {
        "doc_key",
        "sentences",
        "clusters"
    }

    missing_fields = required_fields - set(data.keys())

    if missing_fields:
        errors.append(
            f"Brak wymaganych pól: {sorted(missing_fields)}"
        )

    # ------------------------------------------------------------------
    # 2. sentences
    # ------------------------------------------------------------------

    sentences = data.get("sentences")

    if not isinstance(sentences, list):
        errors.append(
            "'sentences' musi być listą."
        )
        sentences = []

    for sentence_idx, sentence in enumerate(sentences):

        if not isinstance(sentence, list):
            errors.append(
                f"Zdanie {sentence_idx} nie jest listą tokenów."
            )
            continue

        for token_idx, token in enumerate(sentence):

            if not isinstance(token, str):
                errors.append(
                    f"Token [{sentence_idx}][{token_idx}] "
                    f"nie jest tekstem."
                )

    # ------------------------------------------------------------------
    # Liczba tokenów
    # ------------------------------------------------------------------

    total_tokens = sum(
        len(sentence)
        for sentence in sentences
    )

    print(
        f"Liczba tokenów:          {total_tokens}"
    )

    # ------------------------------------------------------------------
    # 3. clusters
    # ------------------------------------------------------------------

    clusters = data.get("clusters")

    if not isinstance(clusters, list):
        errors.append(
            "'clusters' musi być listą."
        )
        clusters = []

    print(
        f"Liczba klastrów:         {len(clusters)}"
    )

    # ------------------------------------------------------------------
    # 4. Kontrola każdego klastra
    # ------------------------------------------------------------------

    all_spans = []
    duplicate_spans = 0
    singleton_clusters = 0

    for cluster_idx, cluster in enumerate(clusters):

        if not isinstance(cluster, list):

            errors.append(
                f"Klaster {cluster_idx} nie jest listą spanów."
            )

            continue

        # --------------------------------------------------------------
        # Singleton
        # --------------------------------------------------------------

        if len(cluster) < 2:

            singleton_clusters += 1

            warnings.append(
                f"Klaster {cluster_idx} jest singletonem."
            )

        # --------------------------------------------------------------
        # Duplikaty spanów
        # --------------------------------------------------------------

        cluster_seen = set()

        for span_idx, span in enumerate(cluster):

            # ----------------------------------------------------------
            # Span musi być listą dwóch elementów
            # ----------------------------------------------------------

            if not isinstance(span, list):

                errors.append(
                    f"Klaster {cluster_idx}, span {span_idx}: "
                    f"span nie jest listą."
                )

                continue

            if len(span) != 2:

                errors.append(
                    f"Klaster {cluster_idx}, span {span_idx}: "
                    f"oczekiwano [start, end], otrzymano {span}."
                )

                continue

            start, end = span

            # ----------------------------------------------------------
            # Indeksy muszą być liczbami całkowitymi
            # ----------------------------------------------------------

            if not isinstance(start, int) or not isinstance(end, int):

                errors.append(
                    f"Klaster {cluster_idx}, span {span_idx}: "
                    f"indeksy nie są liczbami całkowitymi: {span}"
                )

                continue

            # ----------------------------------------------------------
            # start <= end
            # ----------------------------------------------------------

            if start > end:

                errors.append(
                    f"Klaster {cluster_idx}, span {span_idx}: "
                    f"start > end: {span}"
                )

            # ----------------------------------------------------------
            # indeksy >= 0
            # ----------------------------------------------------------

            if start < 0 or end < 0:

                errors.append(
                    f"Klaster {cluster_idx}, span {span_idx}: "
                    f"ujemny indeks: {span}"
                )

            # ----------------------------------------------------------
            # indeksy nie mogą wykraczać poza dokument
            # ----------------------------------------------------------

            if start >= total_tokens or end >= total_tokens:

                errors.append(
                    f"Klaster {cluster_idx}, span {span_idx}: "
                    f"span wychodzi poza dokument: {span}, "
                    f"liczba tokenów={total_tokens}"
                )

            # ----------------------------------------------------------
            # Duplikat
            # ----------------------------------------------------------

            span_tuple = (start, end)

            if span_tuple in cluster_seen:

                duplicate_spans += 1

                errors.append(
                    f"Klaster {cluster_idx}: "
                    f"duplikat spanu {span}."
                )

            cluster_seen.add(span_tuple)

            all_spans.append(
                (
                    start,
                    end,
                    cluster_idx
                )
            )

        # --------------------------------------------------------------
        # 5. Nakładanie się spanów w obrębie klastra
        # --------------------------------------------------------------

        valid_spans = [
            span
            for span in cluster
            if (
                isinstance(span, list)
                and len(span) == 2
                and all(isinstance(x, int) for x in span)
            )
        ]

        valid_spans.sort(
            key=lambda x: (x[0], x[1])
        )

        for i in range(len(valid_spans) - 1):

            current_start, current_end = valid_spans[i]
            next_start, next_end = valid_spans[i + 1]

            if next_start <= current_end:

                errors.append(
                    f"Klaster {cluster_idx}: "
                    f"nakładające się spany "
                    f"{valid_spans[i]} i {valid_spans[i + 1]}."
                )

    # ------------------------------------------------------------------
    # 6. Ta sama wzmianka w różnych klastrach
    # ------------------------------------------------------------------

    span_to_clusters = defaultdict(set)

    for start, end, cluster_idx in all_spans:

        span_to_clusters[
            (start, end)
        ].add(cluster_idx)

    cross_cluster_duplicates = 0

    for span, cluster_ids in span_to_clusters.items():

        if len(cluster_ids) > 1:

            cross_cluster_duplicates += 1

            errors.append(
                f"Wzmianka {list(span)} "
                f"występuje w wielu klastrach: "
                f"{sorted(cluster_ids)}."
            )

    # ------------------------------------------------------------------
    # 7. Maksymalny indeks spanu
    # ------------------------------------------------------------------

    if all_spans:

        max_span_index = max(
            end
            for _, end, _ in all_spans
        )

        expected_max_index = total_tokens - 1

        print(
            f"Najwyższy indeks tokenu: {max_span_index}"
        )

        print(
            f"Oczekiwany maks. indeks: {expected_max_index}"
        )

        if max_span_index > expected_max_index:

            errors.append(
                f"Najwyższy indeks spanu ({max_span_index}) "
                f"przekracza maksymalny indeks dokumentu "
                f"({expected_max_index})."
            )

    # ------------------------------------------------------------------
    # 8. Statystyki
    # ------------------------------------------------------------------

    total_mentions = len(all_spans)

    unique_mentions = len(
        set(
            (start, end)
            for start, end, _ in all_spans
        )
    )

    print(
        f"Liczba wzmianek:         {total_mentions}"
    )

    print(
        f"Unikalnych wzmianek:     {unique_mentions}"
    )

    print(
        f"Singletonów:             {singleton_clusters}"
    )

    print(
        f"Duplikatów spanów:       {duplicate_spans}"
    )

    print(
        f"Wzmianek w wielu klastrach: "
        f"{cross_cluster_duplicates}"
    )

    # ------------------------------------------------------------------
    # 9. Wynik
    # ------------------------------------------------------------------

    print()
    print("-" * 80)

    if warnings:

        print(
            f"OSTRZEŻENIA: {len(warnings)}"
        )

        for warning in warnings[:20]:
            print(
                f"  ! {warning}"
            )

        if len(warnings) > 20:

            print(
                f"  ... oraz {len(warnings) - 20} kolejnych."
            )

    if errors:

        print(
            f"BŁĘDY: {len(errors)}"
        )

        for error in errors[:30]:
            print(
                f"  X {error}"
            )

        if len(errors) > 30:

            print(
                f"  ... oraz {len(errors) - 30} kolejnych."
            )

        print("-" * 80)
        print("KONTROLA JSONL: NIEUDANA")
        print("=" * 80)

        raise ValueError(
            "Wygenerowany JSONL zawiera błędy. "
            "Plik nie powinien zostać zapisany."
        )

    print(
        "KONTROLA JSONL: OK"
    )

    print("-" * 80)

    return True


def convert_webanno(input_path, output_path):
    """
    Konwersja WebAnno TSV 3.x do formatu JSONL używanego przez herference.

    Format anotacji koreferencji:
        *->107-1 *[107]
        *->107-2 36[107]
        *->26-1 57[26]

    Znaczenie:
        *->LINK_ID CLUSTER_ID[CLUSTER_ID]

    Przykład:
        3-1 ... Ustawa *->107-2 36[107]

    oznacza:
        link_id   = 107-2
        cluster   = 107

    Ważne:
    - numeracja tokenów jest globalna dla całego dokumentu,
    - #Text= rozpoczyna nowy segment/zdanie,
    - singletony nie są dodawane do wynikowych klastrów,
    - błędne lub niespójne anotacje są zgłaszane
    """

    print("=" * 80)
    print(f"Konwersja: {input_path}")
    print(f"       -> {output_path}")
    print("=" * 80)

    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Nie znaleziono pliku wejściowego:\n{input_path}"
        )

    # ------------------------------------------------------------------
    # Dane wynikowe
    # ------------------------------------------------------------------

    sentences = []
    current_sentence = []

    mentions_builder = {}

    # Statystyki kontrolne
    total_tokens = 0
    total_annotation_occurrences = 0
    lines_with_coref = 0

    invalid_annotations = 0
    inconsistent_links = 0

    # ------------------------------------------------------------------
    # Regex
    # ------------------------------------------------------------------

    link_pattern = re.compile(
        r"\*->(\d+-\d+)"
    )

    cluster_pattern = re.compile(
        r"(?:\*|\d+)\[(\d+)\]"
    )

    # ------------------------------------------------------------------
    # Czytanie TSV
    # ------------------------------------------------------------------

    with open(input_path, "r", encoding="utf-8") as f:

        for line_no, raw_line in enumerate(f, start=1):

            line = raw_line.rstrip("\r\n")

            # ----------------------------------------------------------
            # Pusta linia
            # ----------------------------------------------------------

            if not line.strip():
                continue

            # ----------------------------------------------------------
            # Linie komentarzy / #Text=
            # ----------------------------------------------------------

            if line.startswith("#"):

                if line.startswith("#Text="):

                    # Każdy #Text= rozpoczyna nowy segment.
                    if current_sentence:
                        sentences.append(current_sentence)
                        current_sentence = []

                continue

            # ----------------------------------------------------------
            # Podział kolumn TSV
            # ----------------------------------------------------------

            parts = line.split("\t")

            if len(parts) < 3:
                print(
                    f"OSTRZEŻENIE: linia {line_no} ma mniej niż "
                    f"3 kolumny. Pomijam:"
                )
                print(f"  {line}")
                continue

            token_id = parts[0]
            token_text = parts[2]

            # Wszystko od kolumny 4 wzwyż traktujemy jako
            # informację o koreferencji.
            annotation_columns = parts[3:]

            coref_info = "\t".join(annotation_columns)

            # Token trafia do aktualnego zdania.
            current_sentence.append(token_text)

            # ----------------------------------------------------------
            # Szukanie anotacji koreferencji
            # ----------------------------------------------------------

            link_ids = link_pattern.findall(coref_info)
            cluster_ids = cluster_pattern.findall(coref_info)

            # Brak koreferencji na tym tokenie.
            if not link_ids:
                total_tokens += 1
                continue

            lines_with_coref += 1

            # ----------------------------------------------------------
            # Kontrola liczby znalezionych elementów
            # ----------------------------------------------------------

            if len(link_ids) != len(cluster_ids):

                invalid_annotations += 1

                print()
                print(
                    f"!!! NIEZGODNOŚĆ w linii {line_no}"
                )
                print(
                    f"Token: {token_id} {token_text}"
                )
                print(
                    f"Coref: {coref_info}"
                )
                print(
                    f"link_ids:    {link_ids}"
                )
                print(
                    f"cluster_ids: {cluster_ids}"
                )

                raise ValueError(
                    f"Nie można jednoznacznie sparsować anotacji "
                    f"koreferencji w linii {line_no}."
                )

            # ----------------------------------------------------------
            # Dodawanie anotacji
            # ----------------------------------------------------------

            for link_id, cluster_id in zip(link_ids, cluster_ids):

                total_annotation_occurrences += 1

                # ------------------------------------------------------
                # Nowa wzmianka
                # ------------------------------------------------------

                if link_id not in mentions_builder:

                    mentions_builder[link_id] = {
                        "tokens": [],
                        "cluster": cluster_id
                    }

                # ------------------------------------------------------
                # Kontrola spójności
                # ------------------------------------------------------

                existing_cluster = mentions_builder[link_id]["cluster"]

                if existing_cluster != cluster_id:

                    inconsistent_links += 1

                    print()
                    print(
                        f"!!! NIESPÓJNY LINK w linii {line_no}"
                    )
                    print(
                        f"link_id: {link_id}"
                    )
                    print(
                        f"Poprzedni klaster: {existing_cluster}"
                    )
                    print(
                        f"Nowy klaster:      {cluster_id}"
                    )
                    print(
                        f"Token: {token_id} {token_text}"
                    )

                    raise ValueError(
                        f"Link {link_id} został przypisany "
                        f"do więcej niż jednego klastra."
                    )

                # ------------------------------------------------------
                # Dodanie tokenu do wzmianki
                # ------------------------------------------------------

                token_index = total_tokens

                mentions_builder[link_id]["tokens"].append(
                    token_index
                )

            total_tokens += 1

    # ------------------------------------------------------------------
    # Ostatnie zdanie / segment
    # ------------------------------------------------------------------

    if current_sentence:
        sentences.append(current_sentence)

    # ------------------------------------------------------------------
    # Budowanie klastrów
    # ------------------------------------------------------------------

    clusters_dict = defaultdict(list)

    for link_id, data in mentions_builder.items():

        tokens = data["tokens"]
        cluster_id = data["cluster"]

        if not tokens:
            print(
                f"OSTRZEŻENIE: wzmianka {link_id} "
                f"nie zawiera żadnych tokenów."
            )
            continue

        start = min(tokens)
        end = max(tokens)

        span = [start, end]

        clusters_dict[cluster_id].append(span)

    # ------------------------------------------------------------------
    # Usunięcie ewentualnych duplikatów spanów
    # ------------------------------------------------------------------

    for cluster_id in clusters_dict:

        unique_spans = set(
            tuple(span)
            for span in clusters_dict[cluster_id]
        )

        clusters_dict[cluster_id] = [
            list(span)
            for span in unique_spans
        ]

        clusters_dict[cluster_id].sort(
            key=lambda span: (span[0], span[1])
        )

    # ------------------------------------------------------------------
    # Kontrola nakładania się wzmianek w obrębie klastra
    # ------------------------------------------------------------------

    overlap_warnings = 0

    for cluster_id, spans in clusters_dict.items():

        for i in range(len(spans)):

            start1, end1 = spans[i]

            for j in range(i + 1, len(spans)):

                start2, end2 = spans[j]

                if start1 <= end2 and start2 <= end1:

                    overlap_warnings += 1

                    print()
                    print(
                        f"OSTRZEŻENIE: nakładające się wzmianki "
                        f"w klastrze {cluster_id}:"
                    )
                    print(
                        f"  {spans[i]}"
                    )
                    print(
                        f"  {spans[j]}"
                    )

    # ------------------------------------------------------------------
    # Usunięcie singletonów
    # ------------------------------------------------------------------

    final_clusters = [
        spans
        for spans in clusters_dict.values()
        if len(spans) > 1
    ]

    # Sortowanie klastrów według pierwszej wzmianki.
    final_clusters.sort(
        key=lambda cluster: (
            cluster[0][0],
            cluster[0][1]
        )
    )

    # ------------------------------------------------------------------
    # Budowa JSONL
    # ------------------------------------------------------------------

    output_data = {
        "doc_key": os.path.basename(input_path),
        "sentences": sentences,
        "clusters": final_clusters
    }

    # Kontrola końcowej struktury JSONL
    validate_jsonl_data(output_data)

    # ------------------------------------------------------------------
    # Kontrola podstawowa
    # ------------------------------------------------------------------

    sentence_token_count = sum(
        len(sentence)
        for sentence in sentences
    )

    if sentence_token_count != total_tokens:

        raise ValueError(
            "Niezgodność liczby tokenów!\n"
            f"Tokeny odczytane z TSV: {total_tokens}\n"
            f"Tokeny zapisane w sentences: {sentence_token_count}"
        )

    # ------------------------------------------------------------------
    # Zapis JSONL
    # ------------------------------------------------------------------

    output_dir = os.path.dirname(output_path)

    if output_dir:
        os.makedirs(
            output_dir,
            exist_ok=True
        )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            json.dumps(
                output_data,
                ensure_ascii=False
            )
            + "\n"
        )

    # ------------------------------------------------------------------
    # Statystyki
    # ------------------------------------------------------------------

    all_mentions = sum(
        len(spans)
        for spans in clusters_dict.values()
    )

    singleton_clusters = sum(
        1
        for spans in clusters_dict.values()
        if len(spans) == 1
    )

    print()
    print("-" * 80)
    print("WYNIK KONWERSJI")
    print("-" * 80)

    print(
        f"Dokument:                 "
        f"{os.path.basename(input_path)}"
    )

    print(
        f"Tokeny:                   "
        f"{total_tokens}"
    )

    print(
        f"Segmenty/zdania:          "
        f"{len(sentences)}"
    )

    print(
        f"Linii z koreferencją:     "
        f"{lines_with_coref}"
    )

    print(
        f"Wystąpień anotacji:       "
        f"{total_annotation_occurrences}"
    )

    print(
        f"Unikalnych wzmianek:      "
        f"{len(mentions_builder)}"
    )

    print(
        f"Klastrów przed filtracją: "
        f"{len(clusters_dict)}"
    )

    print(
        f"Singletonów usuniętych:   "
        f"{singleton_clusters}"
    )

    print(
        f"Klastrów końcowych:       "
        f"{len(final_clusters)}"
    )

    print(
        f"Wzmianek w klastrach:     "
        f"{all_mentions}"
    )

    print(
        f"Ostrzeżeń o overlapach:   "
        f"{overlap_warnings}"
    )

    print(
        f"Niepoprawnych anotacji:   "
        f"{invalid_annotations}"
    )

    print(
        f"Niespójnych linków:       "
        f"{inconsistent_links}"
    )

    print(
        f"Zapisano:                 "
        f"{output_path}"
    )

    print("-" * 80)

convert_webanno("tsv_files/Ustawa_o_ochronie_sygnalistow.docx.tsv", "train/train_uos.jsonl")
convert_webanno("tsv_files/DU_2025_1235-certyfikacja-wykonawcow.docx.tsv", "val/val_cw.jsonl")
convert_webanno("tsv_files/DU_2025_1826-nadzor-nad-ogolnym.docx.tsv", "test/test_nno.jsonl")
convert_webanno("tsv_files/DU_2026_465-dzialalnosc-kosmiczna.docx.tsv", "train/train_dk.jsonl")
convert_webanno("tsv_files/DU_2025_1017-krajowy-system-certyfikacji.docx.tsv", "train/ksc.jsonl")
convert_webanno("tsv_files/DU_2025_1080-szczegolne-zasady-przygotowania.docx.tsv", "train/train_szp.jsonl")
convert_webanno("tsv_files/DU_2025_779-krajowa-siec-kardiologiczna.docx.tsv", "train/train_ksk.jsonl")
convert_webanno("tsv_files/DU_2025_1661-uklady-zbiorowe-pracy.docx.tsv", "train/train_uzp.jsonl")