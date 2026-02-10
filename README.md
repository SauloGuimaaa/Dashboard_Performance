# Dashboard Performance (ETL + Dashboard + Relatório)

Projeto local (Windows) para:
- Ler CSVs brutos em `/input`
- Padronizar/limpar dados (schema alvo)
- Gerar métricas (Top posts, séries diária/semanal, data quality)
- Dashboard interativo (Streamlit + Plotly)
- Exportar tabelas (CSV) e gráficos (PNG via Matplotlib)
- Relatório automático (DOCX + PDF opcional)
- Insights com IA (OpenAI) a partir dos dados filtrados

## Requisitos
- Windows
- Python 3.13 instalado (recomendado)
- (Opcional para PDF) Microsoft Word instalado (para `docx2pdf`)
- (Opcional para IA) Chave OpenAI API para gerar insights

## Rodar com 1 comando
No PowerShell, na raiz do projeto:

```powershell
.\run.ps1
```

## Insights com IA (OpenAI)
No dashboard, há uma seção **Insights com IA** em cada aba. Para usar:
- informe sua OpenAI API Key no campo indicado (ou defina `OPENAI_API_KEY` no ambiente);
- escolha o modelo (ex.: `gpt-4o-mini`);
- ajuste o tamanho da amostra e clique em **Gerar insights**.

Os insights são gerados com base nos filtros aplicados, incluindo categorias de posts e amostra de legendas.
