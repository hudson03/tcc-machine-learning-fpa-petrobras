from __future__ import annotations

import csv
import hashlib
import json
import re
import urllib.request
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parent
ROOT = CODE_DIR.parent
OUT = ROOT / "corpus"
SOURCE_WORKBOOK = ROOT / "dados/DRE_Petrobras_2006_2026_PIPELINE_RECONSTRUIDO_4T18.xlsx"
COMPANY_ID = "25fdf098-34f5-4608-b7fa-17d60b2de47d"
API = (
    "https://apicatalog.mziq.com/filemanager/company/"
    f"{COMPANY_ID}/filter/categories/year/meta"
)
CATEGORY = "central_de_resultados_demonstracoes_financeiras_em_rs"
CENTRAL = "https://www.investidorpetrobras.com.br/resultados-e-comunicados/central-de-resultados/"
ACCESS_DATE = "08 set. 2026"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def request_year(year: int) -> list[dict]:
    body = json.dumps(
        {
            "year": year,
            "categories": [CATEGORY],
            "language": "pt_BR",
            "published": True,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        API, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.load(response)
    if not payload.get("success"):
        raise RuntimeError(f"Consulta sem sucesso para {year}: {payload}")
    return payload["data"]["document_metas"]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for year in range(2006, 2027):
        for document in request_year(year):
            quarter = int(document["file_quarter"])
            direct_url = document.get("link_url") or document.get("permalink")
            match = re.search(r"/([0-9a-f-]{36})(?:\?|$)", direct_url or "", re.I)
            rows.append(
                {
                    "Periodo": f"{quarter}T{str(year)[-2:]}",
                    "Ano": year,
                    "Trimestre": quarter,
                    "Aba_fonte": f"{quarter}T{str(year)[-2:]}",
                    "Titulo_documento": document["file_title"],
                    "Categoria": "Demonstrações Financeiras em R$",
                    "URL_direta": direct_url,
                    "Identificador_documento": match.group(1) if match else "",
                    "URL_central_resultados": CENTRAL,
                    "Data_acesso": ACCESS_DATE,
                }
            )

    rows.sort(key=lambda row: (row["Ano"], row["Trimestre"]))
    keys = [(row["Ano"], row["Trimestre"]) for row in rows]
    expected = [
        (year, quarter)
        for year in range(2006, 2027)
        for quarter in range(1, 5)
        if not (year == 2026 and quarter > 2)
    ]
    if keys != expected:
        raise RuntimeError(
            f"Cobertura documental divergente. Obtido={len(keys)}; esperado={len(expected)}; "
            f"ausentes={sorted(set(expected)-set(keys))}; duplicados={len(keys)-len(set(keys))}"
        )

    csv_path = OUT / "corpus_documental_petrobras_1T06_2T26.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        "generated_on": "2026-09-08",
        "document_count": len(rows),
        "period": "1T06–2T26",
        "coverage_check": {
            "expected": len(expected),
            "observed": len(rows),
            "missing": [],
            "duplicates": 0,
        },
        "catalog_api": API,
        "category_internal_name": CATEGORY,
        "official_results_center": CENTRAL,
        "access_date": ACCESS_DATE,
        "local_source_workbook": {
            "path": str(SOURCE_WORKBOOK.relative_to(ROOT)),
            "sha256": sha256(SOURCE_WORKBOOK),
        },
        "corpus_csv": {
            "path": str(csv_path.relative_to(ROOT)),
            "sha256": sha256(csv_path),
        },
    }
    (OUT / "manifesto_corpus_documental.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
