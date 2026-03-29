# Passo 17: Integrar o Segundo Modelo

## Objetivo

Validar a extensibilidade da arquitetura adicionando um novo backend sem reescrever o pipeline.

## Dependências

- Passo 06 concluído
- Passo 07 concluído
- Passo 09 a 16 concluídos

## Entregáveis

- Segundo runner real
- Registro no registry
- Compatibilidade com inferência, benchmark, comparação e relatório

## Implementação esperada

1. Implementar o segundo modelo respeitando a mesma interface base.
2. Registrar o novo runner no registry.
3. Garantir que toda a aplicação continue operando por configuração, não por condicionais espalhadas.

## Critérios de aceite

- O segundo modelo roda com a mesma CLI.
- A run gerada tem a mesma estrutura.
- Manifesto, métricas e relatórios continuam funcionando sem tratamento especial.

## Como testar

- Executar uma run com o primeiro modelo.
- Executar outra run com o segundo modelo.
- Comparar ambas pelo fluxo já implementado.

## Armadilhas a evitar

- Introduzir exceções arquiteturais só para acomodar o novo runner.
- Mudar o pipeline em vez de adaptar apenas o modelo concreto.
