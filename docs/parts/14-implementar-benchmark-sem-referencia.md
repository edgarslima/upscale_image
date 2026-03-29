# Passo 14: Implementar Benchmark sem Referência

## Objetivo

Adicionar avaliação quantitativa para cenários reais em que não existe imagem alvo de alta resolução.

## Métrica obrigatória

- NIQE

## Dependências

- Passo 13 concluído ou, no mínimo, a infraestrutura de métricas pronta

## Entregáveis

- Execução de métricas no-reference sobre outputs
- Persistência junto ao diretório `metrics/`
- Agregação por rodada

## Implementação esperada

1. Reusar a infraestrutura de persistência de benchmark.
2. Calcular NIQE com `pyiqa`.
3. Distinguir no manifesto e nos relatórios quando a avaliação é full-reference ou no-reference.

## Critérios de aceite

- A aplicação consegue avaliar outputs sem conjunto de referência.
- O resumo agregado deixa claro o tipo de benchmark executado.

## Como testar

- Rodada sem pasta de referência.
- Validação do cálculo e da persistência do NIQE.

## Armadilhas a evitar

- Misturar métricas FR e NR sem indicar o modo de avaliação.
- Tratar ausência de referência como erro fatal quando o modo NR é suficiente.
