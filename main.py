import sys
from pathlib import Path

project_root = Path(r"C:\Users\Tomek\Desktop\magisterka\test_her")
herference_path = project_root / "herference"
if str(herference_path) not in sys.path:
    sys.path.insert(0, str(herference_path))

import spacy
import herference

print("Herference zaimportowane")

nlp = spacy.load("pl_core_news_lg")   # lub md / sm

nlp.add_pipe(
    "herference",
    config={
        "model_name_or_path": "./finetuned_herference",
        "device": "cpu"
    }
)

print("Komponent herference dodany")

text = """Anna Kowalska rozpoczęła pracę w nowej firmie informatycznej w Warszawie. 
Kobieta była bardzo zadowolona z tej zmiany, ponieważ od dawna szukała miejsca, 
w którym mogłaby rozwijać swoje umiejętności programistyczne. 
Pierwszego dnia Anna spotkała swojego przełożonego, Marka Nowaka. 
Mężczyzna oprowadził ją po biurze."""

doc = nlp(text)

print("\n=== Wynik coreferencji ===")

if hasattr(doc._, "coref_clusters"):
    clusters_attr = "coref_clusters"
elif hasattr(doc._, "coref"):
    clusters_attr = "coref"
else:
    clusters_attr = None
    print("Brak atrybutu coref / coref_clusters")

if clusters_attr:
    clusters = getattr(doc._, clusters_attr)
    if clusters:
        for i, cluster in enumerate(clusters):
            mentions = [span.text for span in cluster]
            print(f"Klaster {i+1}: {mentions}")
    else:
        print("Nie znaleziono żadnych klastrów coreferencyjnych.")
else:
    print("Dostępne klucze w doc.spans:", list(doc.spans.keys()))
    for key in doc.spans.keys():
        if "coref" in key.lower():
            print(f"Znaleziono {key}:")
            for group in doc.spans[key]:
                print("  ", [t.text for t in group])