# Dashboard Performance (ETL + Dashboard + Relatório)

Projeto local (Windows) para:
- Ler CSVs brutos em `/input`
- Padronizar/limpar dados (schema alvo)
- Gerar métricas (Top posts, séries diária/semanal, data quality)
- Dashboard interativo (Streamlit + Plotly)
- Exportar tabelas (CSV) e gráficos (PNG via Matplotlib)
- Relatório automático (DOCX + PDF opcional)

## Requisitos
- Windows
- Python 3.13 instalado (recomendado)
- (Opcional para PDF) Microsoft Word instalado (para `docx2pdf`)

## Rodar com 1 comando
No PowerShell, na raiz do projeto:

```powershell
.\run.ps1
