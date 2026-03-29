# Prompt do Passo 12

Execute o passo 12 do projeto: tratamento de erros.

## Leia antes de implementar

- `/home/edgar/dev/upscale_image/docs/parts/12-implementar-tratamento-de-erros.md`
- `/home/edgar/dev/upscale_image/docs/adr/0005-pipeline-deterministico-e-estrategia-de-erros.md`

## Regras fixas

- erro por item: registrar e continuar;
- erro estrutural: abortar cedo;
- tornar a causa da falha clara em CLI, logs e artefatos.

## Antes de finalizar

- validar todos os cenários mínimos documentados;
- atualizar `prompt/control.yaml`.
