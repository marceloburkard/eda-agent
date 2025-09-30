
import streamlit as st
import pandas as pd
from agent_core import load_csv, answer_question, Memory, add_conclusion

st.set_page_config(page_title="EDA Agent", layout="wide")

st.title("EDA Agent (CSV genérico)")

mem = Memory.load()

uploaded = st.file_uploader("Carregue um CSV", type=["csv"])

if uploaded is not None:
    df = load_csv(uploaded)
    st.success(f"Arquivo com {df.shape[0]} linhas e {df.shape[1]} colunas carregado.")
    st.dataframe(df.head(20))

    q = st.text_input("Faça sua pergunta (ex.: 'mostre a matriz de correlação', 'há outliers?', 'distribuição do Amount')")
    if st.button("Responder", use_container_width=True) and q:
        res = answer_question(q, df, mem)
        st.write(res["answer"])
        for name, tbl in res.get("tables", {}).items():
            st.subheader(name)
            st.dataframe(tbl)
        if res.get("fig") is not None:
            st.pyplot(res["fig"])

    sample_note = st.text_area("Observações / Conclusões do Operador (opcional)", "")

    if st.button("Salvar Conclusão", use_container_width=True):
        if sample_note.strip():  # Verificar se há texto para salvar
            add_conclusion(mem, sample_note)
            st.success("Conclusão salva na memória.")
            # Recarregar a memória para mostrar as atualizações
            mem = Memory.load()
            # Limpar o campo de texto após salvar
            st.rerun()
        else:
            st.warning("Por favor, digite uma conclusão antes de salvar.")

    st.subheader("Memória do Agente")
    st.write(mem.facts)

else:
    st.info("Aguardando CSV...")
