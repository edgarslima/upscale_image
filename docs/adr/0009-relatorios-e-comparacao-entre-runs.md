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
- Artefatos derivados relevantes, como resumos de otimização, devem ser serializados de forma que possam ser consumidos por comparação futura.
- O HTML é camada de apresentação; a fonte de verdade continua nos artefatos estruturados.
