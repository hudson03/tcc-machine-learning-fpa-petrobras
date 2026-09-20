# Aprendizado de máquina para previsão de custos operacionais em FP&A

Material complementar do Trabalho de Conclusão de Curso de Hudson Henrique Alves da Silva.

Este repositório reúne o código, a base analítica, o corpus documental e as saídas de auditoria utilizados no estudo de previsão dos custos operacionais trimestrais dos segmentos da Petrobras para horizontes de um a quatro trimestres.

## Conteúdo

- `codigo/`: implementação completa da preparação, modelagem, validação temporal e auditoria;
- `dados/`: planilha conciliada e arquivo estruturado efetivamente utilizados pelos modelos;
- `corpus/`: manifesto dos 82 documentos públicos da Petrobras, de 1T06 a 2T26;
- `fontes_externas/`: série cambial oficial e verificações dos direcionadores econômicos;
- `resultados/`: previsões, métricas, hiperparâmetros, contagens amostrais e testes de estabilidade;
- `reproducao/`: pasta criada pelos scripts para receber uma nova execução, sem sobrescrever os resultados publicados;
- `manifesto_reprodutibilidade.json`: relação dos arquivos, tamanhos e hashes SHA-256.

## Base de dados

A principal base está em `dados/DRE_Petrobras_2006_2026_PIPELINE_RECONSTRUIDO_4T18.xlsx`. Ela foi construída a partir de demonstrações financeiras públicas da Petrobras, conciliada após a mudança de apresentação entre ABAST e RTC no 4T18 e transformada na estrutura analítica usada na modelagem. O arquivo `dados/ml_source.json` contém a extração estruturada consumida pelos scripts.

O escopo geral reúne 621 registros na base analítica e 457 observações segmento-período destinadas à modelagem. Após a criação dos alvos, há 422 observações em H1, 415 em H2, 408 em H3 e 401 em H4.

## Reprodução

Ambiente de referência: Python 3.12.14, NumPy 2.3.5 e openpyxl 3.1.5.

Execute, a partir da raiz do repositório:

```bash
python codigo/run_audit.py
```

A execução completa pode ser demorada, pois repete janelas expansivas para todas as combinações do Random Forest. Os novos arquivos são gravados em `reproducao/`, enquanto `resultados/AUDITORIA_MODELAGEM.md` resume a execução aprovada utilizada no trabalho.

Para instalar as dependências mínimas:

```bash
python -m pip install -r requirements.txt
```

Os scripts `codigo/run_driver_models.py` e `codigo/run_driver_models_sgs3698_full.py` reexecutam os experimentos com Brent e USD/BRL. O segundo consulta a API pública do Banco Central e, portanto, requer conexão com a internet.

O ambiente original não continha scikit-learn. A regressão Ridge foi conferida por uma formulação algébrica independente. As árvores próprias examinam oito limiares quantílicos por variável e, portanto, não são numericamente equivalentes a bibliotecas que avaliam todos os pontos de corte.

## Resultados de controle

- a ampliação do Random Forest para 35, 100, 300 e 500 árvores não alterou a seleção operacional;
- a série USD/BRL contém 246 observações mensais oficiais da SGS 3698, agregadas em 82 trimestres;
- a conferência independente da Ridge apresentou diferença máxima de previsão inferior a 4,1 × 10⁻⁹;
- o agregado previsto é a soma bruta dos segmentos modelados antes das eliminações e não representa o custo consolidado da Petrobras.

## Proveniência

Os dados financeiros e econômicos utilizados são públicos. Os endereços oficiais, identificadores dos documentos e datas de acesso estão registrados em `corpus/corpus_documental_petrobras_1T06_2T26.csv` e `corpus/manifesto_corpus_documental.json`.

O repositório não substitui as fontes oficiais. Para qualquer reutilização, recomenda-se conferir a documentação da Petrobras e do Banco Central do Brasil indicada no corpus.
