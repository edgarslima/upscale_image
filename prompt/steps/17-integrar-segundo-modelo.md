# Prompt do Passo 17

Execute o passo 17 do projeto: integrar o segundo modelo.

## Leia antes de implementar

- `/home/edgar/dev/upscale_image/docs/parts/17-integrar-segundo-modelo.md`
- `/home/edgar/dev/upscale_image/docs/adr/0004-contrato-de-modelo-e-registry.md`

## Regras fixas

- adicionar novo backend sem reescrever pipeline;
- registrar o novo runner no registry;
- preservar compatibilidade com métricas, comparação e relatório.

## Antes de finalizar

- validar execução completa com o novo modelo;
- atualizar `prompt/control.yaml`.
