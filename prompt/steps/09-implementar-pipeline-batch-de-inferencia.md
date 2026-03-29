# Prompt do Passo 09

Execute o passo 09 do projeto: pipeline batch de inferência.

## Leia antes de implementar

- `/home/edgar/dev/upscale_image/docs/parts/09-implementar-pipeline-batch-de-inferencia.md`
- `/home/edgar/dev/upscale_image/docs/passos_execução.md`
- `/home/edgar/dev/upscale_image/docs/adr/0005-pipeline-deterministico-e-estrategia-de-erros.md`

## Regras fixas

- processar diretório inteiro em ordem estável;
- salvar outputs na run atual;
- continuar após falhas recuperáveis por item.

## Antes de finalizar

- validar rodada com sucesso e com falha isolada;
- atualizar `prompt/control.yaml`.
