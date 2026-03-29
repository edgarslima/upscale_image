# Prompt do Passo 21

Execute o passo 21 do projeto: suportar extração de páginas PDF para o pipeline.

## Leia antes de implementar

- `/home/edgar/dev/upscale_image/docs/plano_otimizacao_imagem.md`
- `/home/edgar/dev/upscale_image/docs/parts/21-suportar-extracao-de-paginas-pdf-para-o-pipeline.md`
- `/home/edgar/dev/upscale_image/docs/adr/0002-persistencia-orientada-a-filesystem-e-runs.md`
- `/home/edgar/dev/upscale_image/docs/adr/0003-configuracao-com-precedencia-cli-yaml-defaults.md`
- `/home/edgar/dev/upscale_image/docs/adr/0005-pipeline-deterministico-e-estrategia-de-erros.md`
- `/home/edgar/dev/upscale_image/docs/adr/0006-observabilidade-com-logs-manifesto-e-metricas.md`

## Regras fixas

- tratar PDF apenas quando um parâmetro explícito indicar esse modo;
- persistir páginas extraídas dentro da `run`;
- manter ordenação e numeração determinísticas das páginas;
- preparar páginas para o pipeline existente sem reescrever o núcleo de upscale;
- manter o fluxo atual de imagens intacto.

## Sequência mínima esperada

1. Definir o modo de entrada PDF na configuração ou CLI.
2. Implementar extração de páginas para imagens intermediárias.
3. Integrar a preparação das páginas ao pipeline existente.
4. Registrar metadados de origem PDF no manifesto e nos logs.
5. Cobrir testes de PDFs válidos, inválidos e ordenação das páginas.

## Antes de finalizar

- executar a suíte relevante;
- validar que o fluxo de imagens continua inalterado;
- atualizar `prompt/control.yaml`.
