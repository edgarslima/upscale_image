# Prompt do Passo 08

Execute o passo 08 do projeto: integrar o primeiro runner real.

## Leia antes de implementar

- `/home/edgar/dev/upscale_image/docs/parts/08-integrar-primeiro-runner-real.md`
- `/home/edgar/dev/upscale_image/docs/especificação_tecnica.md`
- `/home/edgar/dev/upscale_image/docs/adr/0004-contrato-de-modelo-e-registry.md`
- `/home/edgar/dev/upscale_image/docs/adr/0007-stack-tecnologica-principal.md`

## Regras fixas

- usar PyTorch para carga, device e inferência;
- suportar CPU e CUDA quando disponível;
- respeitar precisão configurada;
- validar output ampliado de uma imagem real.

## Antes de finalizar

- validar peso, device, precisão e output;
- atualizar `prompt/control.yaml`.
