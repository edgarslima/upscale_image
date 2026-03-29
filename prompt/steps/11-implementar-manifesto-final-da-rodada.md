# Prompt do Passo 11

Execute o passo 11 do projeto: manifesto final da rodada.

## Leia antes de implementar

- `/home/edgar/dev/upscale_image/docs/parts/11-implementar-manifesto-final-da-rodada.md`
- `/home/edgar/dev/upscale_image/docs/adr/0002-persistencia-orientada-a-filesystem-e-runs.md`
- `/home/edgar/dev/upscale_image/docs/adr/0006-observabilidade-com-logs-manifesto-e-metricas.md`

## Regras fixas

- `manifest.json` é artefato obrigatório da run;
- consolidar runtime, modelo, status e timing;
- manter formato estável para comparação futura.

## Antes de finalizar

- validar coerência entre manifesto, outputs e logs;
- atualizar `prompt/control.yaml`.
