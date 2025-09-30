
import os
import json
import io
import math
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest

MEMORY_FILE = "memory.json"

@dataclass
class Memory:
    facts: List[str] = field(default_factory=list)
    qas: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def load(cls, path: str = MEMORY_FILE) -> "Memory":
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return cls(**data)
            except Exception:
                return cls()
        return cls()

    def save(self, path: str = MEMORY_FILE) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"facts": self.facts, "qas": self.qas}, f, ensure_ascii=False, indent=2)

def load_csv(path_or_buffer) -> pd.DataFrame:
    df = pd.read_csv(path_or_buffer)
    return df

def infer_types(df: pd.DataFrame) -> Dict[str, str]:
    types = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            types[col] = "numeric"
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            types[col] = "datetime"
        else:
            # try parse datetimes lightly
            try:
                pd.to_datetime(df[col], errors="raise")
                types[col] = "datetime"
            except Exception:
                types[col] = "categorical"
    return types

def basic_overview(df: pd.DataFrame) -> Dict[str, Any]:
    return {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "columns": list(df.columns),
        "missing_by_col": df.isna().sum().to_dict(),
        "types": infer_types(df),
    }

def summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    return df.describe(include="all").transpose()

def plot_hist(df: pd.DataFrame, col: str) -> plt.Figure:
    fig = plt.figure()
    df[col].dropna().plot(kind="hist", bins=50)
    plt.title(f"Distribuição de {col}")
    plt.xlabel(col)
    plt.ylabel("Frequência")
    return fig

def correlation_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[plt.Figure]]:
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] < 2:
        return pd.DataFrame(), None
    corr = num_df.corr(numeric_only=True)
    fig = plt.figure()
    plt.imshow(corr, interpolation="nearest")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title("Matriz de Correlação")
    plt.colorbar()
    plt.tight_layout()
    return corr, fig

def detect_outliers_iqr(df: pd.DataFrame, col: str) -> Dict[str, Any]:
    s = df[col].dropna()
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    mask = (s < lower) | (s > upper)
    return {"col": col, "lower": float(lower), "upper": float(upper), "count": int(mask.sum())}

def anomalies_isolation_forest(df: pd.DataFrame, n_estimators: int = 200, contamination: float = 0.01) -> pd.Series:
    num_df = df.select_dtypes(include=[np.number]).dropna()
    if num_df.empty:
        return pd.Series([], dtype=int)
    model = IsolationForest(n_estimators=n_estimators, contamination=contamination, random_state=42)
    preds = model.fit_predict(num_df.values)
    # -1 = anomaly
    anomalies_idx = num_df.index[preds == -1]
    return pd.Series(1, index=anomalies_idx)

def kmeans_clusters(df: pd.DataFrame, k: int = 3) -> Tuple[Optional[pd.DataFrame], Optional[plt.Figure]]:
    num_df = df.select_dtypes(include=[np.number]).dropna()
    if num_df.shape[1] < 2 or num_df.shape[0] < k:
        return None, None
    # Reduce to 2D for visualization
    pca = PCA(n_components=2, random_state=42)
    xy = pca.fit_transform(num_df.values)
    km = KMeans(n_clusters=k, n_init="auto", random_state=42)
    labels = km.fit_predict(xy)
    fig = plt.figure()
    for lab in np.unique(labels):
        pts = xy[labels == lab]
        plt.scatter(pts[:,0], pts[:,1], label=f"Cluster {int(lab)}", s=10)
    plt.legend()
    plt.title("Clusters (PCA 2D)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    result = pd.DataFrame({"PC1": xy[:,0], "PC2": xy[:,1], "cluster": labels}, index=num_df.index)
    return result, fig

# Heuristic question router
def answer_question(question: str, df: pd.DataFrame, mem: Memory) -> Dict[str, Any]:
    q = question.lower()
    result: Dict[str, Any] = {"answer": "", "fig": None, "tables": {}}

    if any(k in q for k in ["tipo", "tipos", "types", "colunas", "columns"]):
        info = basic_overview(df)
        result["answer"] = f"{info['rows']} linhas, {info['cols']} colunas.\nTipos inferidos por coluna disponíveis na tabela."
        result["tables"]["types"] = pd.DataFrame(info["types"].items(), columns=["coluna","tipo"])
        return result

    if any(k in q for k in ["distribuição", "histograma", "histogram", "distribution"]):
        # try to find a numeric column named in the question; else first numeric
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        target = None
        for c in num_cols:
            if c.lower() in q:
                target = c
                break
        if not target and num_cols:
            target = num_cols[0]
        if target:
            fig = plot_hist(df, target)
            result["answer"] = f"Histograma de {target}."
            result["fig"] = fig
            return result

    if "correla" in q:
        corr, fig = correlation_matrix(df)
        if fig is not None:
            result["answer"] = "Matriz de correlação entre variáveis numéricas."
            result["tables"]["correlation"] = corr
            result["fig"] = fig
            return result

    if any(k in q for k in ["outlier", "atípico", "anomalia", "anomaly"]):
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if num_cols:
            report = [detect_outliers_iqr(df, c) for c in num_cols[:10]]
            table = pd.DataFrame(report)
            result["answer"] = "Outliers por IQR (amostra de até 10 colunas numéricas)."
            result["tables"]["outliers_iqr"] = table
            # add isolation forest summary
            anomalies = anomalies_isolation_forest(df)
            result["tables"]["anomalies_iforest"] = pd.DataFrame({"is_anomaly": anomalies}).reset_index().rename(columns={"index":"row_index"})
            return result

    if any(k in q for k in ["cluster", "agrupamento"]):
        tbl, fig = kmeans_clusters(df, k=3)
        if fig is not None:
            result["answer"] = "KMeans (k=3) sobre PCA 2D."
            result["tables"]["clusters"] = tbl
            result["fig"] = fig
            return result

    if any(k in q for k in ["média", "mediana", "variância", "desvio", "summary", "describe", "estatíst"]):
        desc = summary_stats(df)
        result["answer"] = "Estatísticas descritivas."
        result["tables"]["summary"] = desc
        return result

    if any(k in q for k in ["tempo", "temporal", "time trend"]):
        # try a "Time" column in seconds
        time_col = None
        for cand in ["Time", "time", "timestamp", "date", "datetime"]:
            if cand in df.columns:
                time_col = cand
                break
        if time_col is not None:
            s = df[time_col]
            if not np.issubdtype(s.dtype, np.number):
                try:
                    s = pd.to_datetime(s)
                    # resample by day
                    series = s.dt.floor("D").value_counts().sort_index()
                except Exception:
                    series = s.value_counts().sort_index()
            else:
                # bucket into hours
                hours = (s/3600.0).astype(float)
                bins = pd.cut(hours, bins=24)
                series = bins.value_counts().sort_index()
            result["answer"] = "Tendência temporal por buckets."
            result["tables"]["temporal"] = series.to_frame("count")
            fig = plt.figure()
            series.plot(kind="line")
            plt.title("Tendência temporal (buckets)")
            result["fig"] = fig
            return result

    # fallback
    desc = summary_stats(df)
    result["answer"] = "Pergunta não mapeada. Mostrando resumo estatístico."
    result["tables"]["summary"] = desc
    return result

def add_conclusion(mem: Memory, text: str) -> None:
    if text and text not in mem.facts:
        mem.facts.append(text)
        mem.save()
