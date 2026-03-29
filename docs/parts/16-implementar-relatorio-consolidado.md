# Passo 16: Implementar Relatório Consolidado

## Objetivo

Expor dados técnicos em um formato legível por humanos, sem substituir os artefatos estruturados.

## Formatos esperados

- JSON
- CSV
- HTML

## Dependências

- Passo 11 concluído
- Passo 15 concluído

## Entregáveis

- Gerador de relatório HTML simples
- Consolidação de parâmetros, tempos, métricas e comparações
- Inclusão de exemplos visuais quando disponível

## Implementação esperada

1. Tratar HTML como camada de leitura, não fonte de verdade.
2. Usar manifesto e métricas persistidas como entrada.
3. Mostrar resumo da run e, quando aplicável, comparação lado a lado.

## Critérios de aceite

- O relatório abre no navegador sem pós-processamento manual.
- O conteúdo principal é compreensível por alguém que não acompanhou a execução ao vivo.
- Os dados apresentados correspondem aos artefatos técnicos persistidos.

## Como testar

- Gerar relatório de uma única run.
- Gerar relatório comparativo entre duas runs.

## Armadilhas a evitar

- Embutir lógica de negócio complexa apenas no HTML.
- Fazer do relatório o único lugar onde certas métricas existem.
- Depender de frontend pesado para uma necessidade essencialmente técnica.
