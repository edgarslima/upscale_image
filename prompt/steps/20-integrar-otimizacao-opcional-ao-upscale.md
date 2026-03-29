# Prompt do Passo 20

Execute o passo 20 do projeto: integrar otimização opcional ao fluxo de `upscale`.

## Leia antes de implementar

- `/home/edgar/dev/upscale_image/docs/plano_otimizacao_imagem.md`
- `/home/edgar/dev/upscale_image/docs/parts/19-implementar-otimizacao-de-imagens-geradas.md`
- `/home/edgar/dev/upscale_image/docs/parts/20-integrar-otimizacao-opcional-ao-upscale.md`
- `/home/edgar/dev/upscale_image/docs/adr/0003-configuracao-com-precedencia-cli-yaml-defaults.md`
- `/home/edgar/dev/upscale_image/docs/adr/0005-pipeline-deterministico-e-estrategia-de-erros.md`
- `/home/edgar/dev/upscale_image/docs/adr/0006-observabilidade-com-logs-manifesto-e-metricas.md`
- `/home/edgar/dev/upscale_image/docs/adr/0010-otimizacao-derivada-e-nao-destrutiva-dos-outputs.md`

## Regras fixas

- manter a otimização desabilitada por padrão no comando `upscale`;
- reutilizar a implementação do passo 19, sem duplicação de lógica;
- executar a otimização apenas depois de a `run` produzir `outputs/*.png`;
- nunca sobrescrever os artefatos canônicos da inferência;
- registrar claramente no log e no manifesto quando a otimização encadeada for usada.

## Sequência mínima esperada

1. Adicionar o parâmetro opcional ao comando `upscale`.
2. Conectar esse parâmetro à camada de configuração efetiva.
3. Encadear a execução da otimização derivada ao final da inferência.
4. Manter compatibilidade integral com o fluxo sem otimização.
5. Cobrir testes do caminho padrão e do caminho com otimização habilitada.

## Antes de finalizar

- executar a suíte relevante;
- validar que `outputs/*.png` permanecem intactos;
- atualizar `prompt/control.yaml`.
