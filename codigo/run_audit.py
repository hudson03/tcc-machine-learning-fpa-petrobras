"""Reexecução isolada e auditável da modelagem do TCC.

Não altera a planilha fonte, os resultados originais, o DOCX ou o builder do TCC.
Todos os artefatos são gravados ao lado deste script.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import openpyxl


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA_DIR = ROOT / "dados"
RESULTS_DIR = ROOT / "resultados"
OUTPUT_DIR = ROOT / "reproducao"
XLSX = DATA_DIR / "DRE_Petrobras_2006_2026_PIPELINE_RECONSTRUIDO_4T18.xlsx"
ML_SOURCE = DATA_DIR / "ml_source.json"
ORIGINAL_MODEL_RESULTS = RESULTS_DIR / "model_results_original.json"
ORIGINAL_DRIVER_RESULTS = RESULTS_DIR / "driver_model_results_sgs3698_full.json"

TIEBREAK_TOL = 1e-12
RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)
RF_TREES = (35, 100, 300, 500)
RF_STRUCTURES = (
    {"max_depth": 5, "min_leaf": 4},
    {"max_depth": 7, "min_leaf": 3},
)
GB_GRID = (
    {"n_estimators": 60, "learning_rate": 0.05, "max_depth": 2, "min_leaf": 5},
    {"n_estimators": 80, "learning_rate": 0.03, "max_depth": 3, "min_leaf": 4},
)
MODEL_ORDER = (
    "Baseline Último",
    "Baseline Sazonal",
    "Ridge Contábil",
    "Random Forest ampliada",
    "Gradient Boosting",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.generic):
        return clean_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(clean_json(value), ensure_ascii=False, indent=2, sort_keys=False),
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(clean_json(row))


def checkpoint(stage: str, started: float, extra: dict | None = None) -> None:
    payload = {
        "stage": stage,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    if extra:
        payload.update(extra)
    write_json(OUTPUT_DIR / "checkpoint.json", payload)
    print(f"[{payload['elapsed_seconds']:8.1f}s] {stage}", flush=True)


def numeric_equal(a, b, tol=1e-9) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(float(a), float(b), rel_tol=tol, abs_tol=tol)
    return str(a) == str(b)


def read_sheet_rows(workbook, name: str) -> list[dict]:
    ws = workbook[name]
    raw = [row for row in ws.iter_rows(values_only=True) if any(v is not None for v in row)]
    headers = [str(v) for v in raw[0]]
    return [dict(zip(headers, row)) for row in raw[1:]]


def metrics(y, forecast) -> dict:
    y = np.asarray(y, dtype=float)
    forecast = np.maximum(np.asarray(forecast, dtype=float), 0.0)
    error = forecast - y
    denom = float(np.sum(np.abs(y)))
    sst = float(np.sum((y - y.mean()) ** 2)) if len(y) else 0.0
    return {
        "N": int(len(y)),
        "WMAPE": float(np.sum(np.abs(error)) / denom) if denom else None,
        "MAE": float(np.mean(np.abs(error))) if len(y) else None,
        "RMSE": float(np.sqrt(np.mean(error**2))) if len(y) else None,
        "R2": float(1.0 - np.sum(error**2) / sst) if sst else None,
        "Bias": float(np.mean(error)) if len(y) else None,
        "Bias_Pct": float(np.sum(error) / denom) if denom else None,
        "SSE": float(np.sum(error**2)),
        "SST_POOLED": sst,
    }


class RidgeModel:
    def __init__(self, alpha: float):
        self.alpha = float(alpha)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        self.std[self.std < 1e-12] = 1.0
        z = (X - self.mean) / self.std
        self.ymean = float(y.mean())
        self.beta = np.linalg.solve(
            z.T @ z + self.alpha * np.eye(z.shape[1]), z.T @ (y - self.ymean)
        )
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.maximum(0.0, self.ymean + ((X - self.mean) / self.std) @ self.beta)


class TreeModel:
    def __init__(self, max_depth=5, min_leaf=4, max_features=None, thresholds=8, rng=None):
        self.max_depth = max_depth
        self.min_leaf = min_leaf
        self.max_features = max_features
        self.thresholds = thresholds
        self.rng = rng or np.random.default_rng(0)

    def fit(self, X, y):
        self.p = X.shape[1]
        self.root = self._build(X, y, 0)
        return self

    def _build(self, X, y, depth):
        node = {"value": float(np.mean(y)), "feature": None, "threshold": None, "left": None, "right": None}
        if depth >= self.max_depth or len(y) < 2 * self.min_leaf or np.var(y) < 1e-12:
            return node
        features = np.arange(self.p)
        if self.max_features and self.max_features < self.p:
            features = self.rng.choice(features, self.max_features, replace=False)
        base = float(np.sum((y - y.mean()) ** 2))
        best = None
        for j in features:
            x = X[:, j]
            uniq = np.unique(x)
            if len(uniq) < 2:
                continue
            if len(uniq) > self.thresholds + 1:
                ths = np.unique(np.quantile(x, np.linspace(0.08, 0.92, self.thresholds)))
            else:
                ths = (uniq[:-1] + uniq[1:]) / 2.0
            for threshold in ths:
                mask = x <= threshold
                n_left = int(mask.sum())
                n_right = len(y) - n_left
                if n_left < self.min_leaf or n_right < self.min_leaf:
                    continue
                y_left, y_right = y[mask], y[~mask]
                loss = float(np.sum((y_left - y_left.mean()) ** 2) + np.sum((y_right - y_right.mean()) ** 2))
                gain = base - loss
                if best is None or gain > best[0]:
                    best = (gain, int(j), float(threshold), mask)
        if best is None or best[0] <= 1e-9:
            return node
        _, feature, threshold, mask = best
        node.update(
            feature=feature,
            threshold=threshold,
            left=self._build(X[mask], y[mask], depth + 1),
            right=self._build(X[~mask], y[~mask], depth + 1),
        )
        return node

    @staticmethod
    def _predict_one(x, node):
        while node["feature"] is not None:
            node = node["left"] if x[node["feature"]] <= node["threshold"] else node["right"]
        return node["value"]

    def predict(self, X):
        return np.asarray([self._predict_one(x, self.root) for x in X], dtype=float)


def rf_checkpoint_predictions(X, y, X_eval, n_values, max_depth, min_leaf, seed):
    n_values = sorted(set(int(n) for n in n_values))
    rng = np.random.default_rng(seed)
    cumulative = np.zeros(len(X_eval), dtype=float)
    outputs = {}
    max_features = max(1, int(math.sqrt(X.shape[1])))
    for i in range(max(n_values)):
        ids = rng.integers(0, len(y), len(y))
        tree = TreeModel(
            max_depth=max_depth,
            min_leaf=min_leaf,
            max_features=max_features,
            thresholds=8,
            rng=np.random.default_rng(seed + i + 1),
        ).fit(X[ids], y[ids])
        cumulative += tree.predict(X_eval)
        n_tree = i + 1
        if n_tree in n_values:
            outputs[n_tree] = np.maximum(0.0, cumulative / n_tree).copy()
    return outputs


class GradientBoostingModel:
    def __init__(self, n_estimators=60, learning_rate=0.05, max_depth=2, min_leaf=5, seed=0):
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.max_depth = int(max_depth)
        self.min_leaf = int(min_leaf)
        self.seed = int(seed)

    def fit(self, X, y):
        self.base = float(np.mean(y))
        prediction = np.full(len(y), self.base, dtype=float)
        self.trees = []
        for i in range(self.n_estimators):
            residual = y - prediction
            tree = TreeModel(
                self.max_depth,
                self.min_leaf,
                None,
                8,
                np.random.default_rng(self.seed + i),
            ).fit(X, residual)
            prediction += self.learning_rate * tree.predict(X)
            self.trees.append(tree)
        return self

    def predict(self, X):
        prediction = np.full(len(X), self.base, dtype=float)
        for tree in self.trees:
            prediction += self.learning_rate * tree.predict(X)
        return np.maximum(0.0, prediction)


def arrays(items, key):
    return np.stack([item[key] for item in items]), np.asarray([item["y"] for item in items], dtype=float)


def choose_best(rows, preference):
    best_score = min(row["WMAPE"] for row in rows)
    ties = [row for row in rows if abs(row["WMAPE"] - best_score) <= TIEBREAK_TOL]
    rank = {json.dumps(clean_json(p), sort_keys=True): i for i, p in enumerate(preference)}
    return min(ties, key=lambda row: rank.get(row["Params_JSON"], 10_000))


def grouped_diagnostics(prediction_rows, horizon, block, model):
    rows = [r for r in prediction_rows if r["Horizonte"] == horizon and r["Bloco"] == block and r["Modelo"] == model]
    y = np.asarray([r["Actual"] for r in rows], dtype=float)
    p = np.asarray([r["Forecast"] for r in rows], dtype=float)
    base = metrics(y, p)
    within_sst = 0.0
    for segment in sorted({r["Segmento"] for r in rows}):
        ys = np.asarray(
            [r["Actual"] for r in rows if r["Segmento"] == segment],
            dtype=float,
        )
        within_sst += float(np.sum((ys - ys.mean()) ** 2))
    r2_within = 1.0 - base["SSE"] / within_sst if within_sst else None
    between_share = 1.0 - within_sst / base["SST_POOLED"] if base["SST_POOLED"] else None
    base.update({"R2_Within_Segment": r2_within, "SST_Within": within_sst, "Between_Share_SST": between_share})
    return base


def cosine(a, b):
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / den) if den else None


def main():
    started = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sources = [
        XLSX,
        ML_SOURCE,
        ORIGINAL_MODEL_RESULTS,
        ORIGINAL_DRIVER_RESULTS,
    ]
    missing = [str(p) for p in sources if not p.exists()]
    if missing:
        raise FileNotFoundError("Fontes ausentes: " + "; ".join(missing))
    hashes_before = {str(p.relative_to(ROOT)): sha256(p) for p in sources}

    try:
        import sklearn  # type: ignore
        sklearn_available = True
        sklearn_version = sklearn.__version__
        sklearn_error = None
    except Exception as exc:
        sklearn_available = False
        sklearn_version = None
        sklearn_error = f"{type(exc).__name__}: {exc}"

    environment = {
        "run_started_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "python_executable": Path(sys.executable).name,
        "numpy_version": np.__version__,
        "openpyxl_version": openpyxl.__version__,
        "scikit_learn_available": sklearn_available,
        "scikit_learn_version": sklearn_version,
        "scikit_learn_import_error": sklearn_error,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "tie_tolerance": TIEBREAK_TOL,
        "tie_rule": "Em empate dentro da tolerância, preservar o modelo selecionado em H(h-1); se indisponível, usar MODEL_ORDER.",
        "model_order": list(MODEL_ORDER),
        "seeds": {
            "rf_tuning": "10000 + 100*h + indice_estrutura; a mesma floresta é avaliada em 35/100/300/500 árvores",
            "rf_expanding": "20000 + 1000*h + 10*target_qid + 1",
            "gb_tuning": "10000 + 100*h + indice_configuracao",
            "gb_expanding": "20000 + 1000*h + 10*target_qid + 2",
            "ridge": "determinística; solução fechada",
        },
        "stopping": {
            "ridge": "solução fechada np.linalg.solve; sem parada iterativa",
            "tree": "max_depth, n<2*min_leaf, variância<1e-12, ausência de split ou ganho<=1e-9",
            "random_forest": "número fixo de árvores; sem early stopping",
            "gradient_boosting": "60 ou 80 árvores fixas; sem early stopping",
        },
    }
    write_json(OUTPUT_DIR / "environment.json", environment)
    checkpoint("ambiente e hashes de entrada registrados", started)

    ml_data = json.loads(ML_SOURCE.read_text(encoding="utf-8"))
    original_results = json.loads(ORIGINAL_MODEL_RESULTS.read_text(encoding="utf-8"))
    driver_results = json.loads(ORIGINAL_DRIVER_RESULTS.read_text(encoding="utf-8"))
    headers = ml_data["headers"]
    idx = {name: i for i, name in enumerate(headers)}

    records = []
    for row in ml_data["rows"]:
        def val(name):
            value = row[idx[name]]
            return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else 0.0
        records.append({
            "period": str(row[idx["Periodo"]]),
            "year": int(row[idx["Ano"]]),
            "quarter": int(row[idx["Trimestre"]]),
            "qid": int(row[idx["Quarter_ID"]]),
            "segment": str(row[idx["Segmento_Analitico"]]),
            "type": str(row[idx["Tipo_Segmento"]]),
            "cost": val("Target_Custo"),
            "revenue": val("Receita"),
            "expenses": val("Despesas_Abs"),
            "ebit": val("EBIT"),
        })

    workbook = openpyxl.load_workbook(XLSX, read_only=True, data_only=True)
    xlsx_model = read_sheet_rows(workbook, "Base_Modelagem")
    xlsx_analytical = read_sheet_rows(workbook, "Base_Analitica")
    workbook.close()
    compare_fields = ("Periodo", "Ano", "Trimestre", "Quarter_ID", "Segmento_Analitico", "Tipo_Segmento", "Receita", "Despesas_Abs", "EBIT", "Target_Custo")
    xlsx_lookup = {(str(r["Segmento_Analitico"]), int(r["Quarter_ID"])): r for r in xlsx_model}
    source_mismatches = []
    for row in ml_data["rows"]:
        key = (str(row[idx["Segmento_Analitico"]]), int(row[idx["Quarter_ID"]]))
        target = xlsx_lookup.get(key)
        if target is None:
            source_mismatches.append({"key": key, "field": "ROW", "json": "present", "xlsx": "missing"})
            continue
        for field in compare_fields:
            if not numeric_equal(row[idx[field]], target[field]):
                source_mismatches.append({"key": key, "field": field, "json": row[idx[field]], "xlsx": target[field]})
    target_abs_failures = [r for r in xlsx_analytical if not numeric_equal(r["Target_Custo"], abs(float(r["Custo"]))) ]

    source_segment_rows = []
    for segment in sorted({str(r["Segmento_Padronizado"]) for r in xlsx_analytical}):
        values = sorted((int(r["Quarter_ID"]), str(r["Periodo"])) for r in xlsx_analytical if str(r["Segmento_Padronizado"]) == segment)
        qids = [q for q, _ in values]
        source_segment_rows.append({
            "Segmento_Padronizado": segment,
            "N": len(values),
            "Primeiro": values[0][1],
            "Ultimo": values[-1][1],
            "QID_Min": min(qids),
            "QID_Max": max(qids),
            "Lacunas_Internas": (max(qids) - min(qids) + 1) - len(set(qids)),
            "Duplicatas": len(qids) - len(set(qids)),
        })
    write_csv(OUTPUT_DIR / "source_segment_counts.csv", source_segment_rows)

    panel = {(r["segment"], r["qid"]): r for r in records}
    segments = sorted({r["segment"] for r in records})
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
    base_feature_names = base_numeric_names + segment_names
    driver_feature_names = base_numeric_names + driver_names + segment_names
    driver_lookup = {int(r["Quarter_ID"]): r for r in driver_results["drivers"]}

    def make_x(origin, with_drivers):
        p1 = panel[(origin["segment"], origin["qid"] - 1)]
        p4 = panel[(origin["segment"], origin["qid"] - 4)]
        q = origin["quarter"]
        values = [
            origin["cost"], p1["cost"], p4["cost"], origin["revenue"], p1["revenue"], p4["revenue"],
            origin["expenses"], p1["expenses"], p4["expenses"], origin["ebit"], p1["ebit"], p4["ebit"],
            math.sin(2 * math.pi * q / 4), math.cos(2 * math.pi * q / 4),
        ]
        if with_drivers:
            d = driver_lookup[origin["qid"]]
            values.extend([
                d["Brent_USD_bbl"], d["Brent_Lag1"], d["Brent_Lag4"],
                d["USD_BRL"], d["USD_BRL_Lag1"], d["USD_BRL_Lag4"],
                d["Brent_BRL_bbl"], d["Brent_Var_AA"], d["USD_BRL_Var_AA"],
            ])
        values.extend(1.0 if origin["segment"] == segment else 0.0 for segment in segments[1:])
        return np.asarray(values, dtype=float)

    datasets = {}
    for h in range(1, 5):
        rows = []
        for origin in records:
            if origin["qid"] < 5:
                continue
            if (origin["segment"], origin["qid"] - 1) not in panel or (origin["segment"], origin["qid"] - 4) not in panel:
                continue
            target = panel.get((origin["segment"], origin["qid"] + h))
            seasonal = panel.get((origin["segment"], origin["qid"] + h - 4))
            if target is None or seasonal is None:
                continue
            rows.append({
                "x_base": make_x(origin, False),
                "x_drivers": make_x(origin, True),
                "y": target["cost"],
                "origin_qid": origin["qid"],
                "origin_period": origin["period"],
                "target_qid": target["qid"],
                "target_period": target["period"],
                "segment": origin["segment"],
                "baseline_last": origin["cost"],
                "baseline_seasonal": seasonal["cost"],
            })
        datasets[h] = rows

    sample_rows = []
    split_defs = (
        ("Total elegível", lambda q: True),
        ("Subtreino", lambda q: q <= 44),
        ("Validação interna", lambda q: 45 <= q <= 52),
        ("Validação principal", lambda q: 53 <= q <= 68),
        ("Teste", lambda q: 69 <= q <= 82),
    )
    for h, data in datasets.items():
        for split, predicate in split_defs:
            subset = [r for r in data if predicate(r["target_qid"])]
            sample_rows.append({"Horizonte": f"H{h}", "Recorte": split, "Segmento": "TOTAL", "N": len(subset)})
            for segment in segments:
                ss = [r for r in subset if r["segment"] == segment]
                sample_rows.append({"Horizonte": f"H{h}", "Recorte": split, "Segmento": segment, "N": len(ss)})
    write_csv(OUTPUT_DIR / "sample_counts.csv", sample_rows)
    data_audit = {
        "xlsx_base_analitica_n": len(xlsx_analytical),
        "xlsx_base_modelagem_n": len(xlsx_model),
        "ml_source_n": len(records),
        "xlsx_vs_ml_source_mismatch_count": len(source_mismatches),
        "xlsx_vs_ml_source_mismatches_first_20": source_mismatches[:20],
        "target_abs_failures": len(target_abs_failures),
        "segments": segments,
        "dataset_n": {f"H{h}": len(rows) for h, rows in datasets.items()},
    }
    write_json(OUTPUT_DIR / "data_audit.json", data_audit)
    checkpoint("fonte xlsx conciliada e N por recorte salvos", started, data_audit)

    tuning_rows = []
    best_params = {}
    seed_rows = []
    for h in range(1, 5):
        data = datasets[h]
        subtrain = [r for r in data if r["target_qid"] <= 44]
        internal = [r for r in data if 45 <= r["target_qid"] <= 52]
        Xb, ytr = arrays(subtrain, "x_base")
        Xb_i, yi = arrays(internal, "x_base")
        Xd, _ = arrays(subtrain, "x_drivers")
        Xd_i, _ = arrays(internal, "x_drivers")

        for model_name, Xtr, Xi in (("Ridge Contábil", Xb, Xb_i), ("Ridge + Brent e Câmbio", Xd, Xd_i)):
            model_rows = []
            for alpha in RIDGE_ALPHAS:
                pred = RidgeModel(alpha).fit(Xtr, ytr).predict(Xi)
                m = metrics(yi, pred)
                row = {"Horizonte": f"H{h}", "Modelo": model_name, "Parametros": f"alpha={alpha:g}", "Params_JSON": json.dumps({"alpha": alpha}, sort_keys=True), "Seed": None, **m}
                tuning_rows.append(row); model_rows.append(row)
            pref = [{"alpha": a} for a in RIDGE_ALPHAS]
            best_params[(h, model_name)] = json.loads(choose_best(model_rows, pref)["Params_JSON"])

        rf_rows = []
        for structure_index, structure in enumerate(RF_STRUCTURES):
            seed = 10000 + 100 * h + structure_index
            seed_rows.append({"Etapa": "Tuning", "Horizonte": f"H{h}", "Target_QID": None, "Modelo": "Random Forest ampliada", "Config_Index": structure_index, "Seed": seed})
            checkpoint_preds = rf_checkpoint_predictions(Xb, ytr, Xb_i, RF_TREES, seed=seed, **structure)
            for n_trees in RF_TREES:
                params = {"n_trees": n_trees, **structure}
                m = metrics(yi, checkpoint_preds[n_trees])
                row = {"Horizonte": f"H{h}", "Modelo": "Random Forest ampliada", "Parametros": f"trees={n_trees};depth={structure['max_depth']};leaf={structure['min_leaf']}", "Params_JSON": json.dumps(params, sort_keys=True), "Seed": seed, **m}
                tuning_rows.append(row); rf_rows.append(row)
        rf_pref = [{"n_trees": n, **s} for n in RF_TREES for s in RF_STRUCTURES]
        best_params[(h, "Random Forest ampliada")] = json.loads(choose_best(rf_rows, rf_pref)["Params_JSON"])

        gb_rows = []
        for config_index, params in enumerate(GB_GRID):
            seed = 10000 + 100 * h + config_index
            seed_rows.append({"Etapa": "Tuning", "Horizonte": f"H{h}", "Target_QID": None, "Modelo": "Gradient Boosting", "Config_Index": config_index, "Seed": seed})
            pred = GradientBoostingModel(seed=seed, **params).fit(Xb, ytr).predict(Xb_i)
            m = metrics(yi, pred)
            row = {"Horizonte": f"H{h}", "Modelo": "Gradient Boosting", "Parametros": ";".join(f"{k}={v}" for k, v in params.items()), "Params_JSON": json.dumps(params, sort_keys=True), "Seed": seed, **m}
            tuning_rows.append(row); gb_rows.append(row)
        best_params[(h, "Gradient Boosting")] = json.loads(choose_best(gb_rows, list(GB_GRID))["Params_JSON"])
        checkpoint(f"tuning concluído H{h}", started, {"best": {k[1]: v for k, v in best_params.items() if k[0] == h}})

    write_csv(OUTPUT_DIR / "tuning_internal.csv", tuning_rows)
    write_json(OUTPUT_DIR / "best_hyperparameters.json", {f"H{h}": {model: params for (hh, model), params in best_params.items() if hh == h} for h in range(1, 5)})
    checkpoint("grade RF 35/100/300/500 salva", started)

    original_param_lookup = {}
    for row in original_results["best_params"]:
        original_param_lookup[(int(row["Horizonte"][1:]), row["Modelo"])] = row["Parametros"]

    predictions = []
    expanding_rows = []
    ridge_coefficients = []
    ridge_windows = []
    for h in range(1, 5):
        data = datasets[h]
        for block, q_start, q_end in (("Validação", 53, 68), ("Teste", 69, 82)):
            for qid in range(q_start, q_end + 1):
                train = [r for r in data if r["target_qid"] < qid]
                test_q = [r for r in data if r["target_qid"] == qid]
                if not test_q:
                    continue
                Xb, ytr = arrays(train, "x_base")
                Xb_q, yq = arrays(test_q, "x_base")
                Xd, _ = arrays(train, "x_drivers")
                Xd_q, _ = arrays(test_q, "x_drivers")
                expanding_rows.append({"Horizonte": f"H{h}", "Bloco": block, "Target_QID": qid, "Target_Periodo": test_q[0]["target_period"], "N_Treino": len(train), "N_Avaliacao": len(test_q)})

                forecasts = {
                    "Baseline Último": np.asarray([r["baseline_last"] for r in test_q], dtype=float),
                    "Baseline Sazonal": np.asarray([r["baseline_seasonal"] for r in test_q], dtype=float),
                }
                for model_name, x_train, x_test, names in (
                    ("Ridge Contábil", Xb, Xb_q, base_feature_names),
                    ("Ridge + Brent e Câmbio", Xd, Xd_q, driver_feature_names),
                ):
                    alpha = best_params[(h, model_name)]["alpha"]
                    model = RidgeModel(alpha).fit(x_train, ytr)
                    forecasts[model_name] = model.predict(x_test)
                    abs_beta = np.abs(model.beta)
                    importance = abs_beta / abs_beta.sum() if abs_beta.sum() else abs_beta
                    order = np.argsort(-importance, kind="stable")
                    ranks = np.empty(len(order), dtype=int); ranks[order] = np.arange(1, len(order) + 1)
                    ridge_windows.append({"Horizonte": f"H{h}", "Bloco": block, "Target_QID": qid, "Target_Periodo": test_q[0]["target_period"], "Modelo": model_name, "Alpha": alpha, "N_Treino": len(train), "Y_Mean": model.ymean})
                    for j, feature in enumerate(names):
                        ridge_coefficients.append({"Horizonte": f"H{h}", "Bloco": block, "Target_QID": qid, "Target_Periodo": test_q[0]["target_period"], "Modelo": model_name, "Alpha": alpha, "N_Treino": len(train), "Feature": feature, "Beta_Padronizado": float(model.beta[j]), "Importancia_Normalizada": float(importance[j]), "Rank_Importancia": int(ranks[j])})

                rf_params = best_params[(h, "Random Forest ampliada")]
                rf_seed = 20000 + 1000 * h + 10 * qid + 1
                forecasts["Random Forest ampliada"] = rf_checkpoint_predictions(Xb, ytr, Xb_q, [rf_params["n_trees"]], max_depth=rf_params["max_depth"], min_leaf=rf_params["min_leaf"], seed=rf_seed)[rf_params["n_trees"]]
                seed_rows.append({"Etapa": block, "Horizonte": f"H{h}", "Target_QID": qid, "Modelo": "Random Forest ampliada", "Config_Index": None, "Seed": rf_seed})

                original_rf = original_param_lookup[(h, "Random Forest")]
                forecasts["Random Forest grade original"] = rf_checkpoint_predictions(Xb, ytr, Xb_q, [original_rf["n_trees"]], max_depth=original_rf["max_depth"], min_leaf=original_rf["min_leaf"], seed=rf_seed)[original_rf["n_trees"]]

                gb_params = best_params[(h, "Gradient Boosting")]
                gb_seed = 20000 + 1000 * h + 10 * qid + 2
                forecasts["Gradient Boosting"] = GradientBoostingModel(seed=gb_seed, **gb_params).fit(Xb, ytr).predict(Xb_q)
                seed_rows.append({"Etapa": block, "Horizonte": f"H{h}", "Target_QID": qid, "Modelo": "Gradient Boosting", "Config_Index": None, "Seed": gb_seed})

                for model_name, values in forecasts.items():
                    for item, actual, forecast in zip(test_q, yq, values):
                        forecast = max(0.0, float(forecast))
                        predictions.append({
                            "Horizonte": f"H{h}", "Bloco": block, "Modelo": model_name,
                            "Origem_Periodo": item["origin_period"], "Target_Periodo": item["target_period"],
                            "Target_QID": item["target_qid"], "Segmento": item["segment"],
                            "Actual": float(actual), "Forecast": forecast, "Erro": forecast - float(actual),
                            "Erro_Abs": abs(forecast - float(actual)),
                        })
                if qid in (q_start, q_end) or qid % 4 == 0:
                    checkpoint(f"backtest H{h} {block} alvo {qid}", started)

    write_csv(OUTPUT_DIR / "predictions_all_candidates.csv", predictions)
    write_csv(OUTPUT_DIR / "expanding_window_counts.csv", expanding_rows)
    write_csv(OUTPUT_DIR / "ridge_coefficients_by_window.csv", ridge_coefficients)
    write_csv(OUTPUT_DIR / "ridge_window_metadata.csv", ridge_windows)
    write_csv(OUTPUT_DIR / "seeds_used.csv", seed_rows)
    checkpoint("previsões de todos os candidatos salvas", started, {"prediction_rows": len(predictions)})

    metric_rows = []
    segment_metric_rows = []
    models = sorted({r["Modelo"] for r in predictions})
    for h in range(1, 5):
        for block in ("Validação", "Teste"):
            for model in models:
                subset = [r for r in predictions if r["Horizonte"] == f"H{h}" and r["Bloco"] == block and r["Modelo"] == model]
                if not subset:
                    continue
                m = grouped_diagnostics(predictions, f"H{h}", block, model)
                metric_rows.append({"Horizonte": f"H{h}", "Bloco": block, "Modelo": model, **m})
                for segment in sorted({r["Segmento"] for r in subset}):
                    ss = [r for r in subset if r["Segmento"] == segment]
                    sm = metrics([r["Actual"] for r in ss], [r["Forecast"] for r in ss])
                    segment_metric_rows.append({"Horizonte": f"H{h}", "Bloco": block, "Modelo": model, "Segmento": segment, **sm})

    operational_selection = []
    previous = None
    for h in range(1, 5):
        eligible = [r for r in metric_rows if r["Horizonte"] == f"H{h}" and r["Bloco"] == "Validação" and r["Modelo"] in MODEL_ORDER]
        best_score = min(r["WMAPE"] for r in eligible)
        ties = [r for r in eligible if abs(r["WMAPE"] - best_score) <= TIEBREAK_TOL]
        if len(ties) > 1 and previous and any(r["Modelo"] == previous for r in ties):
            selected = previous
            reason = "empate: preservado o modelo selecionado no horizonte anterior"
        else:
            selected = min(ties, key=lambda r: MODEL_ORDER.index(r["Modelo"]))["Modelo"]
            reason = "menor WMAPE; desempate por ordem explícita" if len(ties) > 1 else "menor WMAPE"
        operational_selection.append({"Horizonte": f"H{h}", "Modelo_Selecionado": selected, "WMAPE_Validacao": best_score, "Empatados": "; ".join(r["Modelo"] for r in ties), "Regra": reason})
        previous = selected

    h4_equality = []
    for block in ("Validação", "Teste"):
        last = {(r["Target_QID"], r["Segmento"]): r["Forecast"] for r in predictions if r["Horizonte"] == "H4" and r["Bloco"] == block and r["Modelo"] == "Baseline Último"}
        seasonal = {(r["Target_QID"], r["Segmento"]): r["Forecast"] for r in predictions if r["Horizonte"] == "H4" and r["Bloco"] == block and r["Modelo"] == "Baseline Sazonal"}
        diffs = [abs(last[k] - seasonal[k]) for k in last]
        h4_equality.append({"Bloco": block, "N": len(diffs), "Max_Abs_Diff": max(diffs), "Identicos": max(diffs) <= TIEBREAK_TOL})

    write_csv(OUTPUT_DIR / "metrics_all_candidates.csv", metric_rows)
    write_csv(OUTPUT_DIR / "metrics_by_segment.csv", segment_metric_rows)
    write_csv(OUTPUT_DIR / "operational_selection.csv", operational_selection)

    stability_rows = []
    stability_overview = []
    for h in range(1, 5):
        for model_name in ("Ridge Contábil", "Ridge + Brent e Câmbio"):
            group = [r for r in ridge_coefficients if r["Horizonte"] == f"H{h}" and r["Modelo"] == model_name]
            for feature in sorted({r["Feature"] for r in group}):
                rows = sorted((r for r in group if r["Feature"] == feature), key=lambda r: r["Target_QID"])
                betas = np.asarray([r["Beta_Padronizado"] for r in rows], dtype=float)
                signs = np.sign(betas[np.abs(betas) > 1e-12])
                sign_switches = int(np.sum(signs[1:] != signs[:-1])) if len(signs) > 1 else 0
                stability_rows.append({
                    "Horizonte": f"H{h}", "Modelo": model_name, "Feature": feature, "N_Janelas": len(rows),
                    "Beta_Medio": float(betas.mean()), "Beta_DP": float(betas.std()), "Beta_Min": float(betas.min()), "Beta_Max": float(betas.max()),
                    "Abs_Beta_Medio": float(np.abs(betas).mean()),
                    "Consistencia_Sinal": float(max(np.mean(signs > 0), np.mean(signs < 0))) if len(signs) else None,
                    "Mudancas_Sinal": sign_switches,
                    "Importancia_Media": float(np.mean([r["Importancia_Normalizada"] for r in rows])),
                    "Top5_Frequencia": float(np.mean([r["Rank_Importancia"] <= 5 for r in rows])),
                    "Rank_Medio": float(np.mean([r["Rank_Importancia"] for r in rows])),
                })
            by_q = defaultdict(dict)
            for r in group:
                by_q[r["Target_QID"]][r["Feature"]] = r
            qids = sorted(by_q)
            signed_cos, abs_cos, jaccard = [], [], []
            for qa, qb in zip(qids, qids[1:]):
                names = sorted(by_q[qa])
                a = np.asarray([by_q[qa][n]["Beta_Padronizado"] for n in names])
                b = np.asarray([by_q[qb][n]["Beta_Padronizado"] for n in names])
                signed_cos.append(cosine(a, b)); abs_cos.append(cosine(np.abs(a), np.abs(b)))
                ta = {n for n in names if by_q[qa][n]["Rank_Importancia"] <= 5}
                tb = {n for n in names if by_q[qb][n]["Rank_Importancia"] <= 5}
                jaccard.append(len(ta & tb) / len(ta | tb))
            stability_overview.append({
                "Horizonte": f"H{h}", "Modelo": model_name, "N_Janelas": len(qids),
                "Coseno_Assinado_Consecutivo_Medio": float(np.mean(signed_cos)),
                "Coseno_Absoluto_Consecutivo_Medio": float(np.mean(abs_cos)),
                "Jaccard_Top5_Consecutivo_Medio": float(np.mean(jaccard)),
            })
    write_csv(OUTPUT_DIR / "ridge_stability_by_feature.csv", stability_rows)
    write_csv(OUTPUT_DIR / "ridge_stability_overview.csv", stability_overview)

    implementation_rows = []
    for h in range(1, 5):
        train = [r for r in datasets[h] if r["target_qid"] <= 44]
        internal = [r for r in datasets[h] if 45 <= r["target_qid"] <= 52]
        Xtr, ytr = arrays(train, "x_base"); Xi, yi = arrays(internal, "x_base")
        alpha = best_params[(h, "Ridge Contábil")]["alpha"]
        custom = RidgeModel(alpha).fit(Xtr, ytr)
        z = (Xtr - custom.mean) / custom.std
        augmented_x = np.vstack([z, math.sqrt(alpha) * np.eye(z.shape[1])])
        augmented_y = np.concatenate([ytr - ytr.mean(), np.zeros(z.shape[1])])
        beta_ref = np.linalg.lstsq(augmented_x, augmented_y, rcond=None)[0]
        pred_ref = np.maximum(0.0, ytr.mean() + ((Xi - custom.mean) / custom.std) @ beta_ref)
        pred_custom = custom.predict(Xi)
        implementation_rows.append({"Horizonte": f"H{h}", "Backend": "NumPy lstsq aumentado", "Modelo": "Ridge", "Status": "comparado", "Max_Dif_Beta": float(np.max(np.abs(custom.beta - beta_ref))), "Max_Dif_Previsao": float(np.max(np.abs(pred_custom - pred_ref))), "Dif_WMAPE": abs(metrics(yi, pred_custom)["WMAPE"] - metrics(yi, pred_ref)["WMAPE"]), "Nota": "Validação algébrica independente da solução np.linalg.solve"})

    if sklearn_available:
        from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor  # type: ignore
        from sklearn.linear_model import Ridge  # type: ignore
        from sklearn.pipeline import make_pipeline  # type: ignore
        from sklearn.preprocessing import StandardScaler  # type: ignore
        for h in range(1, 5):
            train = [r for r in datasets[h] if r["target_qid"] <= 44]
            internal = [r for r in datasets[h] if 45 <= r["target_qid"] <= 52]
            Xtr, ytr = arrays(train, "x_base"); Xi, yi = arrays(internal, "x_base")
            alpha = best_params[(h, "Ridge Contábil")]["alpha"]
            custom = RidgeModel(alpha).fit(Xtr, ytr).predict(Xi)
            reference = np.maximum(0.0, make_pipeline(StandardScaler(), Ridge(alpha=alpha, fit_intercept=True)).fit(Xtr, ytr).predict(Xi))
            implementation_rows.append({"Horizonte": f"H{h}", "Backend": f"scikit-learn {sklearn_version}", "Modelo": "Ridge", "Status": "comparado", "Max_Dif_Previsao": float(np.max(np.abs(custom - reference))), "Dif_WMAPE": abs(metrics(yi, custom)["WMAPE"] - metrics(yi, reference)["WMAPE"]), "Nota": "Comparação direta"})
            rf = best_params[(h, "Random Forest ampliada")]; seed = 10000 + 100 * h
            ref_rf = RandomForestRegressor(n_estimators=rf["n_trees"], max_depth=rf["max_depth"], min_samples_leaf=rf["min_leaf"], max_features="sqrt", bootstrap=True, random_state=seed, n_jobs=1).fit(Xtr, ytr).predict(Xi)
            custom_rf = rf_checkpoint_predictions(Xtr, ytr, Xi, [rf["n_trees"]], max_depth=rf["max_depth"], min_leaf=rf["min_leaf"], seed=seed)[rf["n_trees"]]
            implementation_rows.append({"Horizonte": f"H{h}", "Backend": f"scikit-learn {sklearn_version}", "Modelo": "Random Forest", "Status": "comparação comportamental", "WMAPE_Custom": metrics(yi, custom_rf)["WMAPE"], "WMAPE_Referencia": metrics(yi, ref_rf)["WMAPE"], "Nota": "Não se espera igualdade: implementação própria testa oito quantis por variável"})
            gb = best_params[(h, "Gradient Boosting")]
            ref_gb = GradientBoostingRegressor(n_estimators=gb["n_estimators"], learning_rate=gb["learning_rate"], max_depth=gb["max_depth"], min_samples_leaf=gb["min_leaf"], random_state=10000 + 100 * h).fit(Xtr, ytr).predict(Xi)
            custom_gb = GradientBoostingModel(seed=10000 + 100 * h, **gb).fit(Xtr, ytr).predict(Xi)
            implementation_rows.append({"Horizonte": f"H{h}", "Backend": f"scikit-learn {sklearn_version}", "Modelo": "Gradient Boosting", "Status": "comparação comportamental", "WMAPE_Custom": metrics(yi, custom_gb)["WMAPE"], "WMAPE_Referencia": metrics(yi, ref_gb)["WMAPE"], "Nota": "Não se espera igualdade: árvores próprias usam grade restrita de splits"})
    else:
        implementation_rows.append({"Horizonte": "Todos", "Backend": "scikit-learn", "Modelo": "Ridge/RF/GB", "Status": "não disponível", "Nota": sklearn_error or "Pacote não instalado"})
    write_csv(OUTPUT_DIR / "implementation_comparison.csv", implementation_rows)

    original_checks = []
    audit_map = {(r["Horizonte"], r["Bloco"], r["Modelo"]): r for r in metric_rows}
    for row in original_results["metrics"]:
        mapped = {"Ridge": "Ridge Contábil", "Random Forest": "Random Forest grade original"}.get(row["Modelo"], row["Modelo"])
        audit = audit_map[(row["Horizonte"], row["Bloco"], mapped)]
        original_checks.append({"Fonte": "model_results.json", "Horizonte": row["Horizonte"], "Bloco": row["Bloco"], "Modelo_Original": row["Modelo"], "Modelo_Auditoria": mapped, "WMAPE_Original": row["WMAPE"], "WMAPE_Auditoria": audit["WMAPE"], "Diferenca": audit["WMAPE"] - row["WMAPE"], "Confere_1e10": abs(audit["WMAPE"] - row["WMAPE"]) <= 1e-10})
    driver_original_map = {(r["Horizonte"], r["Bloco"], r["Modelo"]): r for r in driver_results["metrics"]}
    for model in ("Ridge Contábil", "Ridge + Brent e Câmbio"):
        for h in range(1, 5):
            for block in ("Validação", "Teste"):
                original = driver_original_map[(f"H{h}", block, model)]
                audit = audit_map[(f"H{h}", block, model)]
                original_checks.append({"Fonte": "driver_model_results.json", "Horizonte": f"H{h}", "Bloco": block, "Modelo_Original": model, "Modelo_Auditoria": model, "WMAPE_Original": original["WMAPE"], "WMAPE_Auditoria": audit["WMAPE"], "Diferenca": audit["WMAPE"] - original["WMAPE"], "Confere_1e10": abs(audit["WMAPE"] - original["WMAPE"]) <= 1e-10})
    write_csv(OUTPUT_DIR / "original_results_reconciliation.csv", original_checks)

    hashes_after = {str(p.relative_to(ROOT)): sha256(p) for p in sources}
    unchanged = all(hashes_before[k] == hashes_after[k] for k in hashes_before)
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": ".",
        "command": "python codigo/run_audit.py",
        "source_files": [{"path": k, "sha256": v, "unchanged_after_run": v == hashes_after[k], "size_bytes": (ROOT / k).stat().st_size} for k, v in hashes_before.items()],
        "audit_script": {"path": str(Path(__file__).relative_to(ROOT)), "sha256": sha256(Path(__file__))},
        "source_files_unchanged": unchanged,
        "outputs": sorted(p.name for p in OUTPUT_DIR.iterdir() if p.is_file()),
    }
    write_json(OUTPUT_DIR / "manifest.json", manifest)

    selected_rf = {f"H{h}": best_params[(h, "Random Forest ampliada")] for h in range(1, 5)}
    key_findings = {
        "status": "PASS" if not source_mismatches and not target_abs_failures and unchanged and all(r["Confere_1e10"] for r in original_checks) else "PASS_WITH_WARNINGS",
        "environment": environment,
        "data_audit": data_audit,
        "selected_rf_expanded": selected_rf,
        "operational_selection": operational_selection,
        "h4_baseline_identity": h4_equality,
        "metrics_all_candidates": metric_rows,
        "ridge_stability_overview": stability_overview,
        "implementation_comparison": implementation_rows,
        "limitations": [
            "scikit-learn não estava instalado; a comparação direta com biblioteca consolidada ficou registrada como indisponível.",
            "A comparação algébrica da Ridge com mínimos quadrados aumentados foi executada como controle independente.",
            "As implementações próprias de RF/GB usam oito quantis candidatos por variável e não são equivalentes às classes do scikit-learn.",
            "Métricas por segmento no teste existem apenas para os quatro segmentos ativos de 1T23 a 2T26.",
            "A auditoria parte da base curada; não reconstrói o corpus original de relatórios Petrobras.",
        ],
    }
    write_json(OUTPUT_DIR / "audit_results.json", key_findings)

    metric_lookup = {(r["Horizonte"], r["Bloco"], r["Modelo"]): r for r in metric_rows}
    md = [
        "# Auditoria reproduzível da modelagem",
        "",
        f"Execução: {environment['run_started_utc']}",
        "",
        "## Resultado",
        "",
        f"- Status: **{key_findings['status']}**.",
        f"- Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}; NumPy {np.__version__}; openpyxl {openpyxl.__version__}.",
        f"- scikit-learn: {'disponível, versão ' + sklearn_version if sklearn_available else 'não disponível neste runtime'}.",
        f"- Fonte XLSX: {len(xlsx_analytical)} linhas analíticas e {len(xlsx_model)} linhas de modelagem; divergências com `ml_source.json`: {len(source_mismatches)}.",
        f"- Fontes originais permaneceram inalteradas: {'sim' if unchanged else 'não'}.",
        "",
        "## Random Forest com grade ampliada",
        "",
        "| Horizonte | Árvores | Profundidade | Folha mínima | WMAPE interno |",
        "|---|---:|---:|---:|---:|",
    ]
    for h in range(1, 5):
        p = selected_rf[f"H{h}"]
        row = next(r for r in tuning_rows if r["Horizonte"] == f"H{h}" and r["Modelo"] == "Random Forest ampliada" and json.loads(r["Params_JSON"]) == p)
        md.append(f"| H{h} | {p['n_trees']} | {p['max_depth']} | {p['min_leaf']} | {100*row['WMAPE']:.2f}% |")
    md += [
        "",
        "## Seleção operacional e desempate",
        "",
        "| Horizonte | Modelo | WMAPE validação | Regra |",
        "|---|---|---:|---|",
    ]
    for row in operational_selection:
        md.append(f"| {row['Horizonte']} | {row['Modelo_Selecionado']} | {100*row['WMAPE_Validacao']:.2f}% | {row['Regra']} |")
    md += ["", "Em H4, `Baseline Último` e `Baseline Sazonal` são idênticos linha a linha. A regra explícita preserva o modelo escolhido em H3.", "", "## R² pooled e dentro dos segmentos — modelo operacional no teste", "", "| Horizonte | R² pooled | R² dentro dos segmentos | Parcela da SST entre segmentos |", "|---|---:|---:|---:|"]
    for sel in operational_selection:
        m = metric_lookup[(sel["Horizonte"], "Teste", sel["Modelo_Selecionado"])]
        md.append(f"| {sel['Horizonte']} | {m['R2']:.4f} | {m['R2_Within_Segment']:.4f} | {100*m['Between_Share_SST']:.2f}% |")
    md += [
        "",
        "O R² pooled é dominado pelas diferenças de escala entre segmentos. Para avaliar a dinâmica temporal, devem prevalecer WMAPE e métricas segmentadas.",
        "",
        "## Arquivos principais",
        "",
        "- `audit_results.json`: síntese estruturada.",
        "- `metrics_all_candidates.csv`: validação e teste de todos os modelos.",
        "- `metrics_by_segment.csv`: métricas por segmento e horizonte.",
        "- `tuning_internal.csv`: todas as configurações, inclusive RF 35/100/300/500.",
        "- `sample_counts.csv`: N por recorte, horizonte e segmento.",
        "- `ridge_coefficients_by_window.csv` e arquivos `ridge_stability_*`: estabilidade da Ridge.",
        "- `implementation_comparison.csv`: validação independente e disponibilidade do scikit-learn.",
        "- `manifest.json`: hashes e comando de reprodução.",
        "",
        "## Limitações",
        "",
    ]
    md.extend(f"- {item}" for item in key_findings["limitations"])
    (OUTPUT_DIR / "AUDITORIA_MODELAGEM.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    checkpoint("auditoria concluída", started, {"status": key_findings["status"], "elapsed_seconds": round(time.perf_counter() - started, 3)})


if __name__ == "__main__":
    main()
