# ADR 0009: Relatórios e Comparação entre Runs

## Status

Aceita

## Contexto

Execuções isoladas têm valor limitado sem capacidade de comparação e leitura consolidada.

## Decisão

Ler artefatos persistidos de múltiplas runs para comparação e gerar saídas técnicas em JSON, CSV e HTML.

## Consequências

- O relatório humano depende de manifesto e métricas, não os substitui.
- Comparações devem considerar parâmetros, runtime, tempos e qualidade.
- O HTML é camada de apresentação; a fonte de verdade continua nos artefatos estruturados.
