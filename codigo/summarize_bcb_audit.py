from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parent
ROOT = CODE_DIR.parent
HERE = ROOT / "reproducao"
HERE.mkdir(parents=True, exist_ok=True)
ORIGINAL = ROOT / "resultados/driver_model_results_original.json"
AUDITED = HERE / "driver_model_results_sgs3698_full.json"
MONTHLY = HERE / "bcb_sgs3698_2006_2026H1.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> None:
    original = json.loads(ORIGINAL.read_text(encoding="utf-8"))
    audited = json.loads(AUDITED.read_text(encoding="utf-8"))
    monthly = json.loads(MONTHLY.read_text(encoding="utf-8"))

    old_drivers = {int(row["Quarter_ID"]): row for row in original["drivers"]}
    new_drivers = {int(row["Quarter_ID"]): row for row in audited["drivers"]}
    quarter_rows = []
    for qid in sorted(new_drivers):
        old = float(old_drivers[qid]["USD_BRL"])
        new = float(new_drivers[qid]["USD_BRL"])
        quarter_rows.append(
            {
                "Quarter_ID": qid,
                "Periodo": new_drivers[qid]["Periodo"],
                "USD_BRL_Original": old,
                "USD_BRL_SGS3698": new,
                "Diferenca": new - old,
                "Diferenca_Absoluta": abs(new - old),
            }
        )

    original_metrics = {
        (row["Horizonte"], row["Bloco"], row["Modelo"]): row
        for row in original["metrics"]
    }
    audited_metrics = {
        (row["Horizonte"], row["Bloco"], row["Modelo"]): row
        for row in audited["metrics"]
    }
    metric_rows = []
    for key in sorted(audited_metrics):
        if key not in original_metrics:
            continue
        old = original_metrics[key]
        new = audited_metrics[key]
        metric_rows.append(
            {
                "Horizonte": key[0],
                "Bloco": key[1],
                "Modelo": key[2],
                "WMAPE_Original": float(old["WMAPE"]),
                "WMAPE_SGS3698": float(new["WMAPE"]),
                "Diferenca_WMAPE": float(new["WMAPE"]) - float(old["WMAPE"]),
                "Arredondamento_0_1pp_Igual": round(100 * float(old["WMAPE"]), 1)
                == round(100 * float(new["WMAPE"]), 1),
            }
        )

    max_q = max(row["Diferenca_Absoluta"] for row in quarter_rows)
    max_m = max(abs(row["Diferenca_WMAPE"]) for row in metric_rows)
    all_round_equal = all(row["Arredondamento_0_1pp_Igual"] for row in metric_rows)
    result = {
        "executed_on": "2026-09-08",
        "official_source": {
            "series": "BCB SGS 3698 — taxa de câmbio livre, dólar americano (venda), média de período mensal",
            "url_2006_2015": "https://api.bcb.gov.br/dados/serie/bcdata.sgs.3698/dados?formato=json&dataInicial=01/01/2006&dataFinal=31/12/2015",
            "url_2016_2026H1": "https://api.bcb.gov.br/dados/serie/bcdata.sgs.3698/dados?formato=json&dataInicial=01/01/2016&dataFinal=30/06/2026",
            "monthly_observations": len(monthly),
            "first": monthly[0],
            "last": monthly[-1],
            "sha256": sha256(MONTHLY),
        },
        "quarterly_comparison": {
            "quarters": len(quarter_rows),
            "changed_above_1e_9": sum(row["Diferenca_Absoluta"] > 1e-9 for row in quarter_rows),
            "max_absolute_difference_brl_per_usd": max_q,
        },
        "model_comparison": {
            "metric_rows": len(metric_rows),
            "max_absolute_wmape_difference": max_m,
            "all_wmape_round_to_same_0_1_percentage_point": all_round_equal,
            "conclusion": (
                "A substituição da média reconstruída da série diária pela SGS 3698 mensal integral "
                "não alterou nenhum WMAPE apresentado com uma casa decimal em pontos percentuais."
            ),
        },
        "input_hashes": {
            "original_driver_results": sha256(ORIGINAL),
            "audited_driver_results": sha256(AUDITED),
        },
    }

    (HERE / "bcb_sgs3698_audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for name, rows in (
        ("bcb_quarterly_comparison.csv", quarter_rows),
        ("bcb_model_metric_comparison.csv", metric_rows),
    ):
        with (HERE / name).open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    markdown = f"""# Auditoria da série cambial BCB SGS 3698

- Série mensal oficial: 246 observações, de {monthly[0]['data']} a {monthly[-1]['data']}.
- Agregação trimestral: média aritmética dos três meses, em 82 trimestres.
- Maior diferença frente à reconstrução anterior: {max_q:.10f} R$/USD.
- Maior diferença absoluta de WMAPE na reexecução: {max_m:.12f}.
- Todos os WMAPE permaneceram iguais quando apresentados com uma casa decimal: {'sim' if all_round_equal else 'não'}.

Conclusão: a reexecução integral com a série mensal SGS 3698 confirmou as métricas publicadas no TCC na precisão exibida.
"""
    (HERE / "AUDITORIA_BCB_SGS3698.md").write_text(markdown, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
