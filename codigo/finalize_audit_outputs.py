from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parent
ROOT = CODE_DIR.parent
HERE = ROOT / "reproducao"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> None:
    audit_path = HERE / "audit_results.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    for row in audit["operational_selection"]:
        ties = [name for name in row["Empatados"].split("; ") if name]
        if len(ties) == 1:
            row["Regra"] = "menor WMAPE"

    with (HERE / "operational_selection.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(audit["operational_selection"][0])
        )
        writer.writeheader()
        writer.writerows(audit["operational_selection"])

    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report_path = HERE / "AUDITORIA_MODELAGEM.md"
    report = report_path.read_text(encoding="utf-8")
    report = report.replace(
        "| H2 | Baseline Último | 20.50% | empate: preservado o modelo selecionado no horizonte anterior |",
        "| H2 | Baseline Último | 20.50% | menor WMAPE |",
    ).replace(
        "| H3 | Baseline Último | 25.14% | empate: preservado o modelo selecionado no horizonte anterior |",
        "| H3 | Baseline Último | 25.14% | menor WMAPE |",
    )
    report_path.write_text(report, encoding="utf-8")

    manifest_path = HERE / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["finalized_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["source_files"] = [
        item
        for item in manifest["source_files"]
        if not item["path"].endswith("work\\tcc_build\\build_tcc.py")
    ]
    manifest["audit_script"]["sha256"] = sha256(HERE / "run_audit.py")
    manifest["postprocess_script"] = {
        "path": "codigo/finalize_audit_outputs.py",
        "sha256": sha256(Path(__file__)),
        "purpose": "corrigir apenas o rótulo da regra de seleção quando não houve empate",
    }
    manifest["outputs"] = sorted(path.name for path in HERE.iterdir() if path.is_file())
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit["operational_selection"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
