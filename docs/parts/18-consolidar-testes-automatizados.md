# Passo 18: Consolidar Testes Automatizados

## Objetivo

Estabilizar a base do projeto antes de ampliar escopo, modelos e relatórios.

## Cobertura mínima obrigatória

- parsing de config
- descoberta de arquivos
- criação de run
- resolução de modelo
- geração de manifesto
- execução de pipeline com mock
- lógica de comparação

## Dependências

- Passos anteriores concluídos

## Entregáveis

- Suíte automatizada mínima
- Fixtures pequenas e reprodutíveis
- Cenários críticos protegidos contra regressão

## Implementação esperada

1. Priorizar testes dos contratos mais estáveis.
2. Usar doubles simples quando o modelo real não for necessário.
3. Garantir que a suíte rode de forma reproduzível em ambiente local.

## Critérios de aceite

- A suíte falha quando contratos críticos quebram.
- Os testes cobrem as decisões principais da especificação.
- O projeto consegue evoluir com menor risco de regressão estrutural.

## Como testar

- Executar a suíte completa.
- Confirmar cobertura dos fluxos mínimos.
- Validar cenários negativos relevantes.

## Armadilhas a evitar

- Buscar cobertura alta sem priorização dos pontos de risco.
- Escrever testes frágeis dependentes de hardware específico quando não necessário.
- Deixar sem teste as estruturas persistidas que serão consumidas por outros módulos.
