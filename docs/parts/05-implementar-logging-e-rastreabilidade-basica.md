# Passo 05: Implementar Logging e Rastreabilidade Básica

## Objetivo

Registrar o comportamento da aplicação em console e arquivo para tornar cada run auditável e depurável.

## Dependências

- Passo 04 concluído

## Entregáveis

- Logging em console com Rich
- Logging persistido em arquivo
- Eventos mínimos de início, fim, warnings e falhas
- Registro de configuração efetiva e arquivos descobertos

## Eventos mínimos a registrar

- início da execução
- `run_id`
- modelo selecionado
- device e precisão
- diretório de entrada
- quantidade de tarefas válidas
- warnings de arquivos ignorados
- falhas por item
- resumo final

## Implementação esperada

1. Inicializar logging assim que a run for criada.
2. Garantir dupla escrita: console e `logs.txt`.
3. Padronizar níveis: info, warning, error.
4. Evitar logs soltos e não estruturados durante o loop principal.

## Critérios de aceite

- O operador consegue entender o que aconteceu sem abrir o código.
- O arquivo de log pertence à run e permanece consultável depois.
- Erros recuperáveis e fatais ficam distinguíveis.

## Como testar

- Rodada com imagens válidas.
- Rodada com imagens ignoradas.
- Rodada com falha controlada.

## Armadilhas a evitar

- Logar só no terminal e perder histórico.
- Registrar mensagens importantes apenas dentro de exceções não padronizadas.
- Não incluir contexto suficiente para relacionar log, run e configuração.
