# Prompts de Execução por Etapa

Este diretório foi criado para orquestrar a implementação do projeto com Claude Code CLI passo a passo, usando os documentos em `docs/parts/` e `docs/adr/` como fonte de verdade.

## Objetivo

Permitir que você execute apenas um prompt principal e que o Claude:

- descubra qual foi o último passo executado;
- identifique se existe um passo em andamento para retomar;
- execute o próximo passo válido;
- registre progresso, checkpoints e bloqueios;
- consiga retomar após interrupção no meio do passo.

## Arquivos principais

- [execute_next_step.md](/home/edgar/dev/upscale_image/prompt/execute_next_step.md): prompt principal para pedir ao Claude que execute o próximo passo.
- [protocol.md](/home/edgar/dev/upscale_image/prompt/protocol.md): regras de controle, persistência de estado e retomada.
- [control.yaml](/home/edgar/dev/upscale_image/prompt/control.yaml): estado persistido das etapas.
- [steps/](/home/edgar/dev/upscale_image/prompt/steps): um prompt específico para cada passo.

## Como usar com Claude Code CLI

Exemplo conceitual:

```bash
claude -p "$(cat prompt/execute_next_step.md)"
```

Ou informe ao Claude para executar o conteúdo de [execute_next_step.md](/home/edgar/dev/upscale_image/prompt/execute_next_step.md).

## Regras operacionais

1. O Claude deve sempre ler `prompt/control.yaml` antes de agir.
2. Se existir um passo com status `in_progress`, ele deve retomar esse passo antes de iniciar outro.
3. Se não houver passo em andamento, ele deve escolher o primeiro `pending` em ordem numérica.
4. Antes de editar código, ele deve marcar a etapa como `in_progress` e registrar checkpoint inicial.
5. Após cada marco relevante, ele deve atualizar `prompt/control.yaml`.
6. Ao concluir, ele deve marcar a etapa como `completed`, registrar artefatos, testes executados e preparar o próximo passo como próximo candidato.
7. Se ocorrer bloqueio real, ele deve marcar `blocked` com motivo, último checkpoint e ação recomendada.

## Política de retomada

O estado da retomada é persistido em `prompt/control.yaml` com:

- `status`
- `last_checkpoint`
- `resume_instructions`
- `completed_items`
- `pending_items`
- `touched_files`
- `tests_run`
- `blocking_reason`

Isso permite que uma nova execução do Claude continue de onde parou sem depender apenas do contexto da sessão anterior.
