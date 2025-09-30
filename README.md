# EDA Agent (CSV genérico)

## Como rodar
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Configuração da API OpenAI (para perguntas não mapeadas)

Para usar o fallback com ChatGPT, configure sua API key:

1. Obtenha uma API key em: https://platform.openai.com/api-keys
2. Renomeie o arquivo `env_example.txt` para `.env`
3. Edite o arquivo `.env` e substitua `sua_api_key_aqui` pela sua chave real:
   ```
   OPENAI_API_KEY=sua_api_key_aqui
   ```

O sistema carregará automaticamente as variáveis do arquivo `.env` usando python-dotenv.

## Gerar relatório PDF (opcional)
```bash
python generate_report.py
```
