# Prompt do Passo 04

Execute o passo 04 do projeto: estrutura de rodada.

## Leia antes de implementar

- `/home/edgar/dev/upscale_image/docs/parts/04-implementar-estrutura-de-rodada.md`
- `/home/edgar/dev/upscale_image/docs/adr/0002-persistencia-orientada-a-filesystem-e-runs.md`
- `/home/edgar/dev/upscale_image/docs/adr/0006-observabilidade-com-logs-manifesto-e-metricas.md`

## Regras fixas

- toda execução deve criar `runs/<run_id>/`;
- o `run_id` deve seguir a convenção definida;
- a run deve nascer já preparada para outputs, logs, manifesto e métricas.

## Antes de finalizar

- validar criação de runs distintas sem sobrescrita;
- atualizar `prompt/control.yaml`.
