# Prompt do Passo 19

Execute o passo 19 do projeto: implementar otimização de imagens geradas.

## Leia antes de implementar

- `/home/edgar/dev/upscale_image/docs/plano_otimizacao_imagem.md`
- `/home/edgar/dev/upscale_image/docs/parts/19-implementar-otimizacao-de-imagens-geradas.md`
- `/home/edgar/dev/upscale_image/docs/adr/0002-persistencia-orientada-a-filesystem-e-runs.md`
- `/home/edgar/dev/upscale_image/docs/adr/0003-configuracao-com-precedencia-cli-yaml-defaults.md`
- `/home/edgar/dev/upscale_image/docs/adr/0005-pipeline-deterministico-e-estrategia-de-erros.md`
- `/home/edgar/dev/upscale_image/docs/adr/0006-observabilidade-com-logs-manifesto-e-metricas.md`
- `/home/edgar/dev/upscale_image/docs/adr/0009-relatorios-e-comparacao-entre-runs.md`
- `/home/edgar/dev/upscale_image/docs/adr/0010-otimizacao-derivada-e-nao-destrutiva-dos-outputs.md`

## Regras fixas

- nunca sobrescrever `runs/<run_id>/outputs/*.png`;
- gerar artefatos derivados apenas em `runs/<run_id>/optimized/`;
- persistir `per_image.csv` e `summary.json`;
- registrar a etapa no manifesto e nos logs da `run`;
- tratar falhas por item como recuperáveis;
- preservar compatibilidade com benchmark, comparação e relatório já existentes.

## Sequência mínima esperada

1. Definir o contrato da etapa de otimização e sua configuração efetiva.
2. Implementar processamento determinístico dos PNGs canônicos da `run`.
3. Gerar ao menos variantes `webp` e `jpeg`.
4. Persistir métricas de economia por item e agregadas.
5. Integrar a CLI sem acoplar a otimização ao runner do modelo.
6. Cobrir testes unitários e ao menos um fluxo de integração.

## Antes de finalizar

- executar a suíte relevante;
- validar que os PNGs canônicos permanecem intactos;
- atualizar `prompt/control.yaml`.
