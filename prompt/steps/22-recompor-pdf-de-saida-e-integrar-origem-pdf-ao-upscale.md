# Prompt do Passo 22

Execute o passo 22 do projeto: recompor PDF de saída e integrar origem PDF ao `upscale`.

## Leia antes de implementar

- `/home/edgar/dev/upscale_image/docs/plano_otimizacao_imagem.md`
- `/home/edgar/dev/upscale_image/docs/parts/19-implementar-otimizacao-de-imagens-geradas.md`
- `/home/edgar/dev/upscale_image/docs/parts/20-integrar-otimizacao-opcional-ao-upscale.md`
- `/home/edgar/dev/upscale_image/docs/parts/21-suportar-extracao-de-paginas-pdf-para-o-pipeline.md`
- `/home/edgar/dev/upscale_image/docs/parts/22-recompor-pdf-de-saida-e-integrar-origem-pdf-ao-upscale.md`
- `/home/edgar/dev/upscale_image/docs/adr/0002-persistencia-orientada-a-filesystem-e-runs.md`
- `/home/edgar/dev/upscale_image/docs/adr/0003-configuracao-com-precedencia-cli-yaml-defaults.md`
- `/home/edgar/dev/upscale_image/docs/adr/0005-pipeline-deterministico-e-estrategia-de-erros.md`
- `/home/edgar/dev/upscale_image/docs/adr/0006-observabilidade-com-logs-manifesto-e-metricas.md`
- `/home/edgar/dev/upscale_image/docs/adr/0010-otimizacao-derivada-e-nao-destrutiva-dos-outputs.md`

## Regras fixas

- o modo PDF deve ser explícito no comando;
- o fluxo atual de imagens não pode mudar;
- reutilizar extração PDF e otimização já implementadas, sem duplicação;
- preservar a ordem das páginas ao recompor o PDF;
- manter todos os artefatos intermediários e finais auditáveis na `run`.

## Sequência mínima esperada

1. Adicionar o parâmetro de origem PDF ao comando `upscale`.
2. Encadear extração -> upscale -> otimização opcional -> recomposição do PDF.
3. Persistir o PDF final em diretório dedicado.
4. Atualizar manifesto e logs com origem e artefatos PDF.
5. Cobrir testes do caminho de imagem e do caminho de PDF.

## Antes de finalizar

- executar a suíte relevante;
- validar que a ordem das páginas foi preservada;
- atualizar `prompt/control.yaml`.
