# Auditoria reproduzível da modelagem

Execução: 2026-09-20T21:46:33.488078+00:00

## Resultado

- Status: **PASS**.
- Python 3.12.14; NumPy 2.3.5; openpyxl 3.1.5.
- scikit-learn: não disponível neste runtime.
- Fonte XLSX: 621 linhas analíticas e 457 linhas de modelagem; divergências com `ml_source.json`: 0.
- Fontes originais permaneceram inalteradas: sim.

## Random Forest com grade ampliada

| Horizonte | Árvores | Profundidade | Folha mínima | WMAPE interno |
|---|---:|---:|---:|---:|
| H1 | 100 | 7 | 3 | 11.00% |
| H2 | 500 | 7 | 3 | 12.70% |
| H3 | 35 | 7 | 3 | 11.87% |
| H4 | 100 | 7 | 3 | 14.01% |

## Seleção operacional e desempate

| Horizonte | Modelo | WMAPE validação | Regra |
|---|---|---:|---|
| H1 | Baseline Último | 14.02% | menor WMAPE |
| H2 | Baseline Último | 20.50% | menor WMAPE |
| H3 | Baseline Último | 25.14% | menor WMAPE |
| H4 | Baseline Último | 27.70% | empate: preservado o modelo selecionado no horizonte anterior |

Em H4, `Baseline Último` e `Baseline Sazonal` são idênticos linha a linha. A regra explícita preserva o modelo escolhido em H3.

## R² pooled e dentro dos segmentos — modelo operacional no teste

| Horizonte | R² pooled | R² dentro dos segmentos | Parcela da SST entre segmentos |
|---|---:|---:|---:|
| H1 | 0.9661 | -1.0381 | 98.33% |
| H2 | 0.9612 | -1.3323 | 98.33% |
| H3 | 0.9592 | -1.4519 | 98.33% |
| H4 | 0.9493 | -2.0421 | 98.33% |

O R² pooled é dominado pelas diferenças de escala entre segmentos. Para avaliar a dinâmica temporal, devem prevalecer WMAPE e métricas segmentadas.

## Arquivos principais

- `audit_results.json`: síntese estruturada.
- `metrics_all_candidates.csv`: validação e teste de todos os modelos.
- `metrics_by_segment.csv`: métricas por segmento e horizonte.
- `tuning_internal.csv`: todas as configurações, inclusive RF 35/100/300/500.
- `sample_counts.csv`: N por recorte, horizonte e segmento.
- `ridge_coefficients_by_window.csv` e arquivos `ridge_stability_*`: estabilidade da Ridge.
- `implementation_comparison.csv`: validação independente e disponibilidade do scikit-learn.
- `manifest.json`: hashes e comando de reprodução.

## Limitações

- scikit-learn não estava instalado; a comparação direta com biblioteca consolidada ficou registrada como indisponível.
- A comparação algébrica da Ridge com mínimos quadrados aumentados foi executada como controle independente.
- As implementações próprias de RF/GB usam oito quantis candidatos por variável e não são equivalentes às classes do scikit-learn.
- Métricas por segmento no teste existem apenas para os quatro segmentos ativos de 1T23 a 2T26.
- A auditoria parte da base curada; não reconstrói o corpus original de relatórios Petrobras.
