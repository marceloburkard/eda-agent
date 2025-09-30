
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
import json
import pandas as pd
import os

def build_pdf(output_path="Agentes Autônomos – Relatório da Atividade Extra.pdf"):
    doc = SimpleDocTemplate(output_path, pagesize=A4, leftMargin=2*cm, rightMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Agentes Autônomos – Relatório da Atividade Extra", styles["Title"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("1. Framework escolhida", styles["Heading2"]))
    story.append(Paragraph("Python + Streamlit para UI; pandas/numpy para EDA; scikit-learn para clustering e detecção de anomalias; matplotlib para gráficos; ReportLab para PDF.", styles["BodyText"]))

    story.append(Paragraph("2. Estrutura da Solução", styles["Heading2"]))
    story.append(Paragraph("Arquivos: app.py (UI), agent_core.py (lógica/EDA), memory.json (memória), requirements.txt (deps), este script para gerar PDF.", styles["BodyText"]))

    # Perguntas e respostas de exemplo (puxa da memória se existir)
    story.append(Paragraph("3. Perguntas e Respostas", styles["Heading2"]))
    if os.path.exists("memory.json"):
        with open("memory.json", "r", encoding="utf-8") as f:
            mem = json.load(f)
        facts = mem.get("facts", [])
    else:
        facts = []

    # placeholders (usuário deve rodar o app e preencher)
    examples = [
        ("Tipos de dados e colunas", "O agente listou tipos inferidos e colunas."),
        ("Matriz de correlação", "Correlações fortes foram observadas entre variáveis PCA adjacentes."),
        ("Distribuição de Amount", "A distribuição é altamente assimétrica com cauda longa à direita."),
        ("Outliers", "Foram detectados outliers por IQR e anomalias pelo IsolationForest.")
    ]

    for q, a in examples:
        story.append(Paragraph(f"Pergunta: {q}", styles["BodyText"]))
        story.append(Paragraph(f"Resposta (resumo): {a}", styles["BodyText"]))
        story.append(Spacer(1, 6))

    story.append(Paragraph("4. Conclusões do Agente", styles["Heading2"]))
    if facts:
        for f in facts[:10]:
            story.append(Paragraph(f"- {f}", styles["BodyText"]))
    else:
        story.append(Paragraph("Nenhuma conclusão registrada ainda. Use o app para salvar conclusões.", styles["BodyText"]))

    story.append(Paragraph("5. Códigos fonte", styles["Heading2"]))
    story.append(Paragraph("Anexados junto ao envio (app.py, agent_core.py, generate_report.py, requirements.txt).", styles["BodyText"]))

    story.append(Paragraph("6. Link do agente", styles["Heading2"]))
    story.append(Paragraph("Hospede o app no Streamlit Community Cloud ou similar e inclua aqui.", styles["BodyText"]))

    story.append(Paragraph("7. Observação de segurança", styles["Heading2"]))
    story.append(Paragraph("Chaves e segredos não são necessários nesta solução. Se utilizar integrações, oculte-as.", styles["BodyText"]))

    doc.build(story)
    return output_path

if __name__ == "__main__":
    path = build_pdf()
    print("PDF gerado em:", path)
