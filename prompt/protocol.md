# Protocolo de Execução e Retomada

## Fonte de verdade

O Claude deve seguir, nesta ordem de autoridade:

1. `docs/especificação_tecnica.md`
2. `docs/adr/`
3. `docs/parts/<passo>.md`
4. `CLAUDE.md`
5. `prompt/control.yaml`
6. `prompt/steps/<passo>.md`

Se encontrar conflito entre um prompt e a especificação ou ADR, a especificação e as ADRs vencem.

## Máquina de estado das etapas

Estados permitidos:

- `pending`
- `in_progress`
- `completed`
- `blocked`

## Regra de seleção do passo

1. Ler `prompt/control.yaml`.
2. Se existir um passo `in_progress`, retomar esse passo.
3. Caso contrário, se existir um passo `blocked`, não pular automaticamente. Primeiro validar se o bloqueio ainda existe. Se puder resolver com segurança, continuar a partir dele; se não puder, manter bloqueado e reportar.
4. Caso não exista `in_progress` nem `blocked` retomável, executar o primeiro passo `pending` em ordem.

## Regra de atualização de estado

Antes de implementar:

- definir `status: in_progress`;
- preencher `started_at` se vazio;
- atualizar `last_checkpoint` com a próxima ação concreta;
- preencher `resume_instructions` com um texto curto e operacional.

Durante a implementação:

- atualizar `completed_items`;
- atualizar `pending_items`;
- atualizar `touched_files`;
- atualizar `tests_run`;
- atualizar `last_checkpoint` a cada marco significativo.

Ao concluir:

- definir `status: completed`;
- preencher `completed_at`;
- limpar `blocking_reason`;
- registrar `acceptance_status: passed` ou `partial`;
- registrar artefatos criados e testes executados;
- atualizar `current_step` e `next_step`.

Ao falhar ou ser interrompido:

- manter `status: in_progress` se houver continuação clara;
- usar `blocked` apenas quando existir impedimento real;
- preencher `blocking_reason`;
- registrar exatamente o último ponto consistente concluído;
- escrever `resume_instructions` com o próximo comando ou ação.

## Regra de retomada

Se o passo estiver `in_progress`, o Claude deve:

1. ler `resume_instructions`;
2. inspecionar `touched_files`;
3. verificar quais `completed_items` já estão realmente implementados;
4. continuar somente o que falta em `pending_items`;
5. evitar refazer trabalho que já está consistente.

## Regra de validação

Nenhum passo deve ser marcado como `completed` sem:

- implementação compatível com `docs/parts`;
- verificação dos critérios de aceite;
- registro dos testes ou validações executadas;
- atualização do estado em `prompt/control.yaml`.

## Regra de não desvio

- Não mudar a ordem dos passos sem motivo explícito documentado.
- Não contradizer ADRs.
- Não introduzir arquitetura fora do escopo inicial.
- Não marcar como concluído algo apenas parcialmente iniciado.
