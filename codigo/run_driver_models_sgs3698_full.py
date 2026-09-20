import json
import math
import re
import urllib.request
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EXT = ROOT / "fontes_externas" / "originais"
MODEL_DIR = ROOT / "dados"
AUDIT_DIR = ROOT / "reproducao"
AUDIT_DIR.mkdir(parents=True, exist_ok=True)


def clean_tokens(text):
    return [token.strip() for token in re.split(r"[\r\n]+", text) if token.strip()]


def parse_eia_monthly():
    rows = json.loads((EXT / "eia_brent_monthly_rows.json").read_text(encoding="utf-8"))
    observations = {}
    for row in rows:
        tokens = clean_tokens(row)
        if not tokens:
            continue
        year_token = tokens[0].replace("\xa0", "").strip()
        if not re.fullmatch(r"20\d{2}", year_token):
            continue
        year = int(year_token)
        values = []
        for token in tokens[1:]:
            if re.fullmatch(r"\d+(?:\.\d+)?", token):
                values.append(float(token))
        for month, value in enumerate(values[:12], start=1):
            observations[(year, month)] = value
    return observations


MONTHS_PT = {"jan": 1, "fev": 2, "mar": 3, "abr": 4, "mai": 5, "jun": 6,
             "jul": 7, "ago": 8, "set": 9, "out": 10, "nov": 11, "dez": 12}


def parse_bcb_monthly_pages():
    urls = (
        "https://api.bcb.gov.br/dados/serie/bcdata.sgs.3698/dados?formato=json&dataInicial=01/01/2006&dataFinal=31/12/2015",
        "https://api.bcb.gov.br/dados/serie/bcdata.sgs.3698/dados?formato=json&dataInicial=01/01/2016&dataFinal=30/06/2026",
    )
    observations = {}
    raw_rows = []
    for url in urls:
        with urllib.request.urlopen(url, timeout=30) as response:
            rows = json.load(response)
        raw_rows.extend(rows)
        for row in rows:
            dt = datetime.strptime(row["data"], "%d/%m/%Y")
            observations[(dt.year, dt.month)] = float(row["valor"])
    (AUDIT_DIR / "bcb_sgs3698_2006_2026H1.json").write_text(
        json.dumps(raw_rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return observations


def parse_bcb_daily_fallback():
    return {}


def quarter_id(year, quarter):
    return (year - 2006) * 4 + quarter


def build_quarterly_drivers():
    brent = parse_eia_monthly()
    fx_sgs = parse_bcb_monthly_pages()
    fx_daily = parse_bcb_daily_fallback()
    fx = dict(fx_daily)
    fx.update(fx_sgs)
    rows = []
    for year in range(2006, 2027):
        for quarter in range(1, 5):
            qid = quarter_id(year, quarter)
            if qid > 82:
                continue
            months = range((quarter - 1) * 3 + 1, quarter * 3 + 1)
            brent_vals = [brent.get((year, month)) for month in months]
            fx_vals = [fx.get((year, month)) for month in months]
            if any(value is None for value in brent_vals + fx_vals):
                raise ValueError(f"Série mensal incompleta para {quarter}T{year}: Brent={brent_vals}, câmbio={fx_vals}")
            brent_q = float(np.mean(brent_vals))
            fx_q = float(np.mean(fx_vals))
            rows.append({
                "Periodo": f"{quarter}T{str(year)[-2:]}",
                "Ano": year,
                "Trimestre": quarter,
                "Quarter_ID": qid,
                "Brent_USD_bbl": brent_q,
                "USD_BRL": fx_q,
                "Brent_BRL_bbl": brent_q * fx_q,
                "Meses_Brent": 3,
                "Meses_USD_BRL": 3,
            })
    lookup = {row["Quarter_ID"]: row for row in rows}
    for row in rows:
        qid = row["Quarter_ID"]
        for suffix, lag in (("Lag1", 1), ("Lag4", 4)):
            prior = lookup.get(qid - lag)
            row[f"Brent_{suffix}"] = prior["Brent_USD_bbl"] if prior else None
            row[f"USD_BRL_{suffix}"] = prior["USD_BRL"] if prior else None
        prior4 = lookup.get(qid - 4)
        row["Brent_Var_AA"] = row["Brent_USD_bbl"] / prior4["Brent_USD_bbl"] - 1 if prior4 else None
        row["USD_BRL_Var_AA"] = row["USD_BRL"] / prior4["USD_BRL"] - 1 if prior4 else None
    return rows


source = json.loads((MODEL_DIR / "ml_source.json").read_text(encoding="utf-8"))
headers = source["headers"]
idx = {name: i for i, name in enumerate(headers)}


def val(row, name):
    value = row[idx[name]]
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else 0.0


records = []
for row in source["rows"]:
    records.append({
        "period": str(row[idx["Periodo"]]),
        "year": int(row[idx["Ano"]]),
        "quarter": int(row[idx["Trimestre"]]),
        "qid": int(row[idx["Quarter_ID"]]),
        "segment": str(row[idx["Segmento_Analitico"]]),
        "type": str(row[idx["Tipo_Segmento"]]),
        "cost": val(row, "Target_Custo"),
        "revenue": val(row, "Receita"),
        "expenses": val(row, "Despesas_Abs"),
        "ebit": val(row, "EBIT"),
    })

panel = {(row["segment"], row["qid"]): row for row in records}
segments = sorted({row["segment"] for row in records})
drivers = build_quarterly_drivers()
driver_lookup = {row["Quarter_ID"]: row for row in drivers}

base_numeric_names = [
    "Custo_t", "Custo_t_1", "Custo_t_4", "Receita_t", "Receita_t_1", "Receita_t_4",
    "Despesas_t", "Despesas_t_1", "Despesas_t_4", "EBIT_t", "EBIT_t_1", "EBIT_t_4",
    "Trimestre_Sin", "Trimestre_Cos",
]
driver_names = [
    "Brent_t", "Brent_t_1", "Brent_t_4", "USD_BRL_t", "USD_BRL_t_1", "USD_BRL_t_4",
    "Brent_BRL_t", "Brent_Var_AA", "USD_BRL_Var_AA",
]
segment_names = [f"Segmento={segment}" for segment in segments[1:]]


def base_values(origin):
    p1 = panel[(origin["segment"], origin["qid"] - 1)]
    p4 = panel[(origin["segment"], origin["qid"] - 4)]
    q = origin["quarter"]
    return [
        origin["cost"], p1["cost"], p4["cost"], origin["revenue"], p1["revenue"], p4["revenue"],
        origin["expenses"], p1["expenses"], p4["expenses"], origin["ebit"], p1["ebit"], p4["ebit"],
        math.sin(2 * math.pi * q / 4), math.cos(2 * math.pi * q / 4),
    ]


def make_x(origin, with_drivers):
    values = base_values(origin)
    if with_drivers:
        d = driver_lookup[origin["qid"]]
        values.extend([
            d["Brent_USD_bbl"], d["Brent_Lag1"], d["Brent_Lag4"],
            d["USD_BRL"], d["USD_BRL_Lag1"], d["USD_BRL_Lag4"],
            d["Brent_BRL_bbl"], d["Brent_Var_AA"], d["USD_BRL_Var_AA"],
        ])
    values.extend([1.0 if origin["segment"] == segment else 0.0 for segment in segments[1:]])
    return np.asarray(values, dtype=float)


datasets = {}
for horizon in range(1, 5):
    items = []
    for origin in records:
        if origin["qid"] < 5 or origin["qid"] not in driver_lookup:
            continue
        if (origin["segment"], origin["qid"] - 1) not in panel or (origin["segment"], origin["qid"] - 4) not in panel:
            continue
        target = panel.get((origin["segment"], origin["qid"] + horizon))
        if not target:
            continue
        seasonal = panel.get((origin["segment"], origin["qid"] + horizon - 4))
        items.append({
            "x_base": make_x(origin, False),
            "x_drivers": make_x(origin, True),
            "y": target["cost"],
            "origin_qid": origin["qid"],
            "origin_period": origin["period"],
            "target_qid": target["qid"],
            "target_period": target["period"],
            "segment": origin["segment"],
            "baseline_last": origin["cost"],
            "baseline_seasonal": seasonal["cost"] if seasonal else np.nan,
        })
    datasets[horizon] = items


def arrays(items, feature_key):
    return np.stack([item[feature_key] for item in items]), np.asarray([item["y"] for item in items], dtype=float)


def metrics(y, forecast):
    y = np.asarray(y, dtype=float)
    forecast = np.maximum(np.asarray(forecast, dtype=float), 0)
    error = forecast - y
    denominator = np.sum(np.abs(y))
    sst = np.sum((y - y.mean()) ** 2)
    return {
        "N": int(len(y)),
        "WMAPE": float(np.sum(np.abs(error)) / denominator) if denominator else None,
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(error ** 2))),
        "R2": float(1 - np.sum(error ** 2) / sst) if sst else None,
        "Bias": float(np.mean(error)),
        "Bias_Pct": float(np.sum(error) / denominator) if denominator else None,
    }


class RidgeModel:
    def __init__(self, alpha):
        self.alpha = alpha

    def fit(self, X, y):
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        self.std[self.std < 1e-12] = 1.0
        z = (X - self.mean) / self.std
        self.ymean = float(np.mean(y))
        self.beta = np.linalg.solve(z.T @ z + self.alpha * np.eye(z.shape[1]), z.T @ (y - self.ymean))
        importance = np.abs(self.beta)
        self.importance = importance / importance.sum() if importance.sum() else importance
        return self

    def predict(self, X):
        return np.maximum(0, self.ymean + ((X - self.mean) / self.std) @ self.beta)


alpha_grid = (0.1, 1.0, 10.0, 100.0, 1000.0)
model_specs = {
    "Ridge Contábil": "x_base",
    "Ridge + Brent e Câmbio": "x_drivers",
}
best_params = {}
tuning = []
all_metrics = []
predictions = []
importance = []
selection = []

for horizon in range(1, 5):
    dataset = datasets[horizon]
    subtrain = [item for item in dataset if item["target_qid"] <= 44]
    internal = [item for item in dataset if 45 <= item["target_qid"] <= 52]
    for model_name, feature_key in model_specs.items():
        x_train, y_train = arrays(subtrain, feature_key)
        x_internal, y_internal = arrays(internal, feature_key)
        scored = []
        for alpha in alpha_grid:
            model = RidgeModel(alpha).fit(x_train, y_train)
            score = metrics(y_internal, model.predict(x_internal))
            scored.append((score["WMAPE"], alpha))
            tuning.append({"Horizonte": f"H{horizon}", "Modelo": model_name, "Alpha": alpha,
                           "WMAPE_Interno": score["WMAPE"], "N_Interno": score["N"]})
        best_params[(horizon, model_name)] = min(scored, key=lambda item: item[0])[1]

    for block, q_start, q_end in (("Validação", 53, 68), ("Teste", 69, 82)):
        actual = []
        metadata = []
        forecasts = {name: [] for name in list(model_specs) + ["Baseline Último", "Baseline Sazonal"]}
        for qid in range(q_start, q_end + 1):
            train = [item for item in dataset if item["target_qid"] < qid]
            test_q = [item for item in dataset if item["target_qid"] == qid]
            if not test_q:
                continue
            actual.extend([item["y"] for item in test_q])
            metadata.extend(test_q)
            for model_name, feature_key in model_specs.items():
                x_train, y_train = arrays(train, feature_key)
                x_test, _ = arrays(test_q, feature_key)
                model = RidgeModel(best_params[(horizon, model_name)]).fit(x_train, y_train)
                forecasts[model_name].extend(model.predict(x_test).tolist())
            forecasts["Baseline Último"].extend([item["baseline_last"] for item in test_q])
            forecasts["Baseline Sazonal"].extend([item["baseline_seasonal"] for item in test_q])

        for model_name, values in forecasts.items():
            summary = metrics(actual, values)
            all_metrics.append({"Horizonte": f"H{horizon}", "Bloco": block, "Modelo": model_name, **summary})
            for item, actual_value, forecast_value in zip(metadata, actual, values):
                forecast_value = max(0, float(forecast_value))
                predictions.append({
                    "Horizonte": f"H{horizon}", "Bloco": block, "Modelo": model_name,
                    "Origem_Periodo": item["origin_period"], "Target_Periodo": item["target_period"],
                    "Target_Quarter_ID": item["target_qid"], "Segmento": item["segment"],
                    "Actual": actual_value, "Forecast": forecast_value,
                    "Erro": forecast_value - actual_value, "Erro_Abs": abs(forecast_value - actual_value),
                })

    validation_metrics = [row for row in all_metrics if row["Horizonte"] == f"H{horizon}" and row["Bloco"] == "Validação"]
    best_operational = min(validation_metrics, key=lambda row: row["WMAPE"])["Modelo"]
    best_ml = min([row for row in validation_metrics if row["Modelo"] in model_specs], key=lambda row: row["WMAPE"])["Modelo"]
    for kind, model_name in (("Modelo operacional", best_operational), ("Melhor Ridge", best_ml)):
        validation = next(row for row in all_metrics if row["Horizonte"] == f"H{horizon}" and row["Bloco"] == "Validação" and row["Modelo"] == model_name)
        test = next(row for row in all_metrics if row["Horizonte"] == f"H{horizon}" and row["Bloco"] == "Teste" and row["Modelo"] == model_name)
        selection.append({
            "Horizonte": f"H{horizon}", "Tipo": kind, "Modelo": model_name,
            "Alpha": best_params.get((horizon, model_name)),
            "WMAPE_Validacao": validation["WMAPE"], "WMAPE_Teste": test["WMAPE"],
            "MAE_Teste": test["MAE"], "RMSE_Teste": test["RMSE"], "R2_Teste": test["R2"],
            "Bias_Pct_Teste": test["Bias_Pct"],
        })

    for model_name, feature_key in model_specs.items():
        x_all, y_all = arrays(dataset, feature_key)
        final_model = RidgeModel(best_params[(horizon, model_name)]).fit(x_all, y_all)
        names = base_numeric_names + (driver_names if feature_key == "x_drivers" else []) + segment_names
        for feature, weight in sorted(zip(names, final_model.importance), key=lambda item: item[1], reverse=True):
            importance.append({"Horizonte": f"H{horizon}", "Modelo": model_name, "Driver": feature, "Importancia": float(weight)})


aggregated = []
groups = defaultdict(lambda: {"Actual": 0.0, "Forecast": 0.0, "Segmentos": 0})
for row in predictions:
    if row["Bloco"] != "Teste":
        continue
    key = (row["Horizonte"], row["Modelo"], row["Target_Quarter_ID"], row["Target_Periodo"])
    groups[key]["Actual"] += row["Actual"]
    groups[key]["Forecast"] += row["Forecast"]
    groups[key]["Segmentos"] += 1
for (horizon, model_name, qid, period), values in sorted(groups.items()):
    aggregated.append({"Horizonte": horizon, "Modelo": model_name, "Target_Quarter_ID": qid,
                       "Target_Periodo": period, **values})

result = {
    "metadata": {
        "generated_at": "2026-09-08",
        "method": "Previsão direta multi-horizonte com janela expansiva",
        "leakage_control": "Drivers econômicos observados apenas no trimestre de origem t; alvo em t+h",
        "driver_sources": {
            "Brent": "U.S. EIA - Europe Brent Spot Price FOB, média mensal em USD/barril",
            "USD_BRL": "Banco Central do Brasil, SGS 3698, série mensal integral consultada via API em duas janelas",
        },
        "base_features": base_numeric_names + segment_names,
        "driver_features": driver_names,
        "segments": segments,
    },
    "drivers": drivers,
    "tuning": tuning,
    "metrics": all_metrics,
    "selection": selection,
    "predictions": predictions,
    "aggregated_test": aggregated,
    "importance": importance,
}

(AUDIT_DIR / "driver_model_results_sgs3698_full.json").write_text(
    json.dumps(result, ensure_ascii=False), encoding="utf-8"
)
print(json.dumps({
    "driver_rows": len(drivers),
    "first_driver": drivers[0],
    "last_driver": drivers[-1],
    "selection": selection,
    "test_metrics": [row for row in all_metrics if row["Bloco"] == "Teste"],
}, ensure_ascii=False, indent=2))
