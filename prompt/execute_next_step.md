# Executar Próximo Passo

Leia e siga rigorosamente estes arquivos antes de agir:

- `/home/edgar/dev/upscale_image/prompt/protocol.md`
- `/home/edgar/dev/upscale_image/prompt/control.yaml`
- `/home/edgar/dev/upscale_image/CLAUDE.md`
- `/home/edgar/dev/upscale_image/docs/especificação_tecnica.md`
- `/home/edgar/dev/upscale_image/docs/adr/README.md`

Sua missão é executar exatamente um passo do plano de implementação do projeto.

## Regras obrigatórias

1. Verifique `prompt/control.yaml`.
2. Se houver algum passo com `status: in_progress`, retome esse passo.
3. Se não houver passo em andamento, escolha o primeiro passo `pending` em ordem numérica.
4. Antes de editar qualquer arquivo, atualize `prompt/control.yaml` para marcar o passo como `in_progress` e registre checkpoint inicial.
5. Abra e siga o prompt específico do passo em `prompt/steps/`.
6. Consulte também o arquivo correspondente em `docs/parts/`.
7. Consulte as ADRs relevantes antes de implementar.
8. Execute a implementação completa do passo, sem avançar para o passo seguinte.
9. Valide os critérios de aceite do passo.
10. Atualize `prompt/control.yaml` ao final com:

- status final do passo;
- arquivos alterados;
- testes rodados;
- itens concluídos;
- itens pendentes;
- instrução de retomada;
- próximo passo sugerido.

## Regras de retomada

Se o passo já estiver `in_progress`, você deve retomar de onde parou usando:

- `last_checkpoint`
- `resume_instructions`
- `completed_items`
- `pending_items`
- `touched_files`

Não reinicie o passo do zero sem necessidade.

## Resultado esperado

Ao final desta execução, apenas uma destas condições pode ser verdadeira:

- o passo atual ficou `completed`;
- o passo atual ficou `blocked` com motivo explícito;
- o passo atual permaneceu `in_progress` com checkpoint preciso para retomada.
