# Prompt do Passo 23

Execute o passo 23 do projeto: otimizar páginas antes da recomposição do PDF final.

## Leia antes de implementar

- `/home/edgar/dev/upscale_image/docs/plano_otimizacao_imagem.md`
- `/home/edgar/dev/upscale_image/docs/parts/19-implementar-otimizacao-de-imagens-geradas.md`
- `/home/edgar/dev/upscale_image/docs/parts/22-recompor-pdf-de-saida-e-integrar-origem-pdf-ao-upscale.md`
- `/home/edgar/dev/upscale_image/docs/parts/23-otimizar-paginas-antes-da-recomposicao-do-pdf-final.md`
- `/home/edgar/dev/upscale_image/docs/adr/0002-persistencia-orientada-a-filesystem-e-runs.md`
- `/home/edgar/dev/upscale_image/docs/adr/0003-configuracao-com-precedencia-cli-yaml-defaults.md`
- `/home/edgar/dev/upscale_image/docs/adr/0005-pipeline-deterministico-e-estrategia-de-erros.md`
- `/home/edgar/dev/upscale_image/docs/adr/0006-observabilidade-com-logs-manifesto-e-metricas.md`
- `/home/edgar/dev/upscale_image/docs/adr/0010-otimizacao-derivada-e-nao-destrutiva-dos-outputs.md`

## Regras fixas

- não recompor o PDF final diretamente a partir dos PNGs canônicos de upscale;
- preparar páginas específicas para composição final em diretório dedicado;
- usar orçamento padrão de tamanho final igual a `2x` o PDF original, salvo override explícito;
- registrar no manifesto e nos logs o alvo de tamanho e o resultado final;
- não mascarar como sucesso normal um PDF final que continue fora do orçamento.

## Sequência mínima esperada

1. Introduzir `compose_ready_pages/` como nova camada pré-composição.
2. Definir orçamento de tamanho do PDF final.
3. Implementar compressão progressiva das páginas antes da recomposição.
4. Fazer o compositor usar as páginas preparadas.
5. Cobrir testes de sucesso dentro do orçamento e falha/degradação fora do orçamento.

## Antes de finalizar

- executar a suíte relevante;
- validar que os `outputs/*.png` canônicos permanecem intactos;
- atualizar `prompt/control.yaml`.
