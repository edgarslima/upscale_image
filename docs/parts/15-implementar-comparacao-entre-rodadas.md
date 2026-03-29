# Passo 15: Implementar Comparação entre Rodadas

## Objetivo

Ler execuções já concluídas e transformar resultados isolados em base real de decisão técnica.

## Dependências

- Passo 11 concluído
- Passo 13 concluído
- Passo 14 concluído

## Entregáveis

- Leitura de múltiplas runs
- Consolidação de manifestos
- Consolidação de métricas
- Cálculo de deltas

## Implementação esperada

1. Ler `manifest.json` e arquivos em `metrics/`.
2. Extrair parâmetros comparáveis: modelo, escala, device, precisão, tile e tempos.
3. Comparar totais, médias, estabilidade e qualidade.
4. Gerar uma saída consolidada serializável.

## Critérios de aceite

- A comparação evidencia diferenças entre execuções.
- Os números comparados são rastreáveis até os artefatos originais.
- O formato de saída pode alimentar relatório humano depois.

## Como testar

- Gerar pelo menos duas runs distintas.
- Validar se os deltas batem com os artefatos originais.

## Armadilhas a evitar

- Comparar runs com campos incompatíveis sem indicar a limitação.
- Recalcular métricas quando bastaria ler artefatos persistidos.
- Ignorar parâmetros de runtime que explicam diferenças de desempenho.
