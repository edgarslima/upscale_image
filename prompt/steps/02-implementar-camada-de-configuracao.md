# Prompt do Passo 02

Execute o passo 02 do projeto: implementar a camada de configuração.

## Leia antes de implementar

- `/home/edgar/dev/upscale_image/docs/parts/02-implementar-camada-de-configuracao.md`
- `/home/edgar/dev/upscale_image/docs/especificação_tecnica.md`
- `/home/edgar/dev/upscale_image/docs/adr/0003-configuracao-com-precedencia-cli-yaml-defaults.md`

## Regras fixas

- respeitar `CLI > YAML > defaults`;
- produzir um objeto de configuração canônico;
- validar cedo parâmetros inconsistentes.

## Antes de finalizar

- validar merge, sobrescrita e casos inválidos;
- atualizar `prompt/control.yaml` com checkpoint, testes e artefatos.
