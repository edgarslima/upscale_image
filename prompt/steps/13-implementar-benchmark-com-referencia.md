# Prompt do Passo 13

Execute o passo 13 do projeto: benchmark com referência.

## Leia antes de implementar

- `/home/edgar/dev/upscale_image/docs/parts/13-implementar-benchmark-com-referencia.md`
- `/home/edgar/dev/upscale_image/docs/adr/0008-avaliacao-de-qualidade-fr-e-nr.md`
- `/home/edgar/dev/upscale_image/docs/adr/0006-observabilidade-com-logs-manifesto-e-metricas.md`

## Regras fixas

- calcular PSNR, SSIM e LPIPS;
- persistir `per_image.csv` e `summary.json`;
- tratar pareamento ausente ou inconsistente com clareza.

## Antes de finalizar

- validar conjunto pareado pequeno;
- atualizar `prompt/control.yaml`.
