
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
import openai
from dotenv import load_dotenv

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

# Palavras‑chave e rotas mapeadas para intents padronizados
# Cada intent possui: lista de palavras‑chave já suportadas e um token de injeção
KEYWORD_ROUTES: Dict[str, Dict[str, Any]] = {
    "types": {
        "keywords": ["tipo", "tipos", "types", "colunas", "columns"],
        "inject": "types",
    },
    "distribution": {
        "keywords": ["distribuição", "histograma", "histogram", "distribution"],
        "inject": "distribuição",
    },
    "correlation": {
        "keywords": ["correla"],
        "inject": "correla",
    },
    "outliers": {
        "keywords": ["outlier", "atípico", "anomalia", "anomaly"],
        "inject": "outlier",
    },
    "clusters": {
        "keywords": ["cluster", "agrupamento"],
        "inject": "cluster",
    },
    "summary": {
        "keywords": ["média", "mediana", "variância", "desvio", "summary", "describe", "estatíst"],
        "inject": "summary",
    },
    "temporal": {
        "keywords": ["tempo", "temporal", "time trend"],
        "inject": "tempo",
    },
}


def llm_classify_and_rewrite(question: str, df: pd.DataFrame, mem: Memory) -> Tuple[Optional[str], str]:
    """
    Usa a OpenAI (gpt-3.5-turbo) para:
    1) Classificar a intenção do usuário segundo intents conhecidas
    2) Reescrever a pergunta, se necessário
    Retorna (intent|None, pergunta_reescrita)
    """
    try:
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None, question

        client = openai.OpenAI(api_key=api_key)

        allowed_intents = list(KEYWORD_ROUTES.keys())
        memory_snippet = "\n".join(f"- {fact}" for fact in mem.facts[:10]) if mem and mem.facts else "(vazio)"

        basic_info = basic_overview(df)
        json_example = json.dumps({
            "intent": "types|distribution|correlation|outliers|clusters|summary|temporal|none",
            "rewritten_question": "...",
        }, ensure_ascii=False)
        prompt = (
            "Você é um roteador de intenções.\n"
            "Classifique a pergunta do usuário em UMA das intents pré-definidas ou 'none' se não corresponder.\n"
            "Se necessário, reescreva a pergunta para ficar mais clara e objetiva.\n\n"
            f"INTENTS PERMITIDAS: {allowed_intents}\n"
            f"FORMATO DE SAÍDA STRICT JSON (sem texto extra): {json_example}\n\n"
            f"FATOS DA MEMÓRIA (contexto):\n{memory_snippet}\n\n"
            f"COLUNAS DO DATASET: {', '.join(basic_info['columns'][:50])}\n\n"
            f"PERGUNTA ORIGINAL: {question}"
        )

        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Você responde apenas JSON válido quando solicitado."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=300,
        )
        content = resp.choices[0].message.content or "{}"
        data = json.loads(content)
        intent = data.get("intent")
        rewritten = data.get("rewritten_question") or question
        if intent not in KEYWORD_ROUTES and intent != "none":
            # Intent desconhecida → tratar como none
            intent = None
        if intent == "none":
            intent = None
        return intent, rewritten
    except Exception:
        # Em caso de qualquer erro, mantém pergunta original e sem intent
        return None, question


# Heuristic question router (com roteamento por LLM antes das heurísticas)
def answer_question(question: str, df: pd.DataFrame, mem: Memory) -> Dict[str, Any]:
    # Primeiro tenta classificar/reescrever via LLM
    intent, rewritten_question = llm_classify_and_rewrite(question, df, mem)
    if rewritten_question:
        question = rewritten_question

    # Se houver uma intent reconhecida, injeta um token correspondente
    if intent and intent in KEYWORD_ROUTES:
        inject_token = KEYWORD_ROUTES[intent]["inject"]
        question = f"{question} {inject_token}"

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

    # fallback - usar OpenAI ChatGPT
    return call_openai_chatgpt(question, df, mem)

def add_conclusion(mem: Memory, text: str) -> None:
    if text and text not in mem.facts:
        mem.facts.append(text)
        mem.save()

def call_openai_chatgpt(question: str, df: pd.DataFrame, mem: Memory) -> Dict[str, Any]:
    """
    Chama a API da OpenAI ChatGPT para responder perguntas não mapeadas
    """
    try:
        # Carregar variáveis do arquivo .env
        load_dotenv()
        
        # Configurar a API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return {
                "answer": "Erro: OPENAI_API_KEY não configurada. Configure a variável no arquivo .env.",
                "tables": {},
                "fig": None
            }
        
        client = openai.OpenAI(api_key=api_key)
        
        # Preparar informações do dataset
        basic_info = basic_overview(df)
        summary_stats_df = summary_stats(df)
        
        # Preparar fatos da memória para incluir no prompt
        memory_context = ""
        if mem.facts:
            memory_context = f"""
FATOS E CONCLUSÕES ANTERIORES SALVAS NA MEMÓRIA:
{chr(10).join(f"- {fact}" for fact in mem.facts)}

"""
        
        # Criar prompt estruturado
        prompt = f"""
Você é um assistente especializado em análise de dados. Analise o seguinte dataset e responda à pergunta do usuário.

{memory_context}INFORMAÇÕES DO DATASET:
- Número de linhas: {basic_info['rows']}
- Número de colunas: {basic_info['cols']}
- Colunas: {', '.join(basic_info['columns'])}
- Tipos de dados: {basic_info['types']}
- Valores faltantes por coluna: {basic_info['missing_by_col']}

ESTATÍSTICAS DESCRITIVAS:
{summary_stats_df.to_string()}

PERGUNTA DO USUÁRIO: {question}

INSTRUÇÕES:
1. Analise os dados fornecidos e responda à pergunta de forma clara e objetiva
2. Considere os fatos e conclusões anteriores da memória ao formular sua resposta
3. Se possível, sugira análises adicionais que poderiam ser úteis
4. Identifique padrões, tendências ou insights interessantes nos dados
5. Se a pergunta não puder ser respondida com os dados disponíveis, explique o porquê
6. Responda em português brasileiro

RESPOSTA:
"""
        
        # Chamar a API
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Você é um especialista em análise de dados e estatística."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.7
        )
        
        answer = response.choices[0].message.content
        
        # Tentar gerar visualizações baseadas na resposta
        result = {
            "answer": answer,
            "tables": {"summary": summary_stats_df},
            "fig": None
        }
        
        # Se a pergunta menciona distribuição ou histograma, tentar gerar
        if any(word in question.lower() for word in ["distribuição", "histograma", "histogram", "distribution"]):
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if num_cols:
                target_col = None
                for col in num_cols:
                    if col.lower() in question.lower():
                        target_col = col
                        break
                if not target_col:
                    target_col = num_cols[0]
                
                try:
                    fig = plot_hist(df, target_col)
                    result["fig"] = fig
                except Exception:
                    pass
        
        # Se a pergunta menciona correlação, tentar gerar matriz
        elif "correla" in question.lower():
            try:
                corr, fig = correlation_matrix(df)
                if fig is not None:
                    result["tables"]["correlation"] = corr
                    result["fig"] = fig
            except Exception:
                pass
        
        return result
        
    except Exception as e:
        # Fallback para o comportamento original em caso de erro
        desc = summary_stats(df)
        return {
            "answer": f"Erro ao processar pergunta com IA: {str(e)}. Mostrando resumo estatístico.",
            "tables": {"summary": desc},
            "fig": None
        }
