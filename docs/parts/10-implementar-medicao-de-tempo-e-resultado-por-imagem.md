# Passo 10: Implementar Medição de Tempo e Resultado por Imagem

## Objetivo

Tornar a execução mensurável em nível de item e preparar a agregação da rodada.

## Dependências

- Passo 09 concluído

## Entregáveis

- Medição de tempo por imagem
- Registro de dimensões de entrada e saída
- Status individual padronizado
- Estrutura de agregação parcial

## Dados por item recomendados

- `input_path`
- `output_path`
- `status`
- `error_message`
- `input_width`
- `input_height`
- `output_width`
- `output_height`
- `inference_time_ms`

## Implementação esperada

1. Cronometrar apenas a parte de inferência e, se útil, também o total por item.
2. Registrar metadados estruturados e não apenas em log textual.
3. Manter a mesma estrutura para sucesso e falha.

## Critérios de aceite

- Cada item processado gera um registro técnico.
- A rodada já consegue calcular total, média e taxa de sucesso.
- Os dados ficam prontos para manifesto e benchmark.

## Como testar

- Rodada pequena com pelo menos um sucesso e uma falha.
- Validação manual das dimensões e tempos registrados.

## Armadilhas a evitar

- Medir tempo de forma inconsistente entre itens.
- Não registrar falhas com a mesma estrutura de dados dos sucessos.
- Deixar os resultados apenas em memória sem persistência planejada.
