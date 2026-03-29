# Passo 23: Otimizar Páginas Antes da Recomposição do PDF Final

## Objetivo

Reduzir agressivamente o tamanho do PDF reconstruído, otimizando e compactando as páginas processadas antes da composição final em `pdf/rebuilt/`, com meta padrão de manter o arquivo final em no máximo `2x` o tamanho do PDF original.

## ADRs relacionadas

- [0002-persistencia-orientada-a-filesystem-e-runs.md](/home/edgar/dev/upscale_image/docs/adr/0002-persistencia-orientada-a-filesystem-e-runs.md)
- [0003-configuracao-com-precedencia-cli-yaml-defaults.md](/home/edgar/dev/upscale_image/docs/adr/0003-configuracao-com-precedencia-cli-yaml-defaults.md)
- [0005-pipeline-deterministico-e-estrategia-de-erros.md](/home/edgar/dev/upscale_image/docs/adr/0005-pipeline-deterministico-e-estrategia-de-erros.md)
- [0006-observabilidade-com-logs-manifesto-e-metricas.md](/home/edgar/dev/upscale_image/docs/adr/0006-observabilidade-com-logs-manifesto-e-metricas.md)
- [0010-otimizacao-derivada-e-nao-destrutiva-dos-outputs.md](/home/edgar/dev/upscale_image/docs/adr/0010-otimizacao-derivada-e-nao-destrutiva-dos-outputs.md)

## Problema a resolver

O fluxo atual de PDF pode gerar páginas ampliadas muito pesadas, fazendo com que o PDF final reconstruído fique inviável para abertura, transferência e uso cotidiano. Já houve caso em que um PDF original de 21 páginas com cerca de `28 MB` resultou em um arquivo final de `2.1 GB`, o que é operacionalmente inútil. O sistema precisa otimizar esse estágio ao extremo antes da recomposição, adotando um orçamento objetivo de tamanho final.

## Dependências

- Passo 19 concluído
- Passo 20 concluído
- Passo 22 concluído

## Entregáveis

- Nova etapa pré-composição para preparar páginas otimizadas para PDF
- Orçamento de tamanho do PDF final baseado no tamanho do original
- Estratégia progressiva de compressão por página
- Diretório `pdf/compose_ready_pages/` dentro da `run`
- Manifesto e logs com tamanho original, orçamento alvo, tamanho final e razão obtida
- Testes cobrindo sucesso dentro do orçamento e falha quando o alvo não puder ser atingido

## Escopo obrigatório

- Criar uma camada intermediária entre `outputs/` e `pdf/rebuilt/`
- Preparar páginas específicas para composição do PDF, sem alterar os `outputs/*.png` canônicos
- Usar por padrão um alvo de tamanho final igual a `2x` o tamanho do PDF original
- Permitir sobrescrita desse alvo por parâmetro ou configuração, mantendo o padrão conservador
- Aplicar compressão progressiva antes da composição final:
  qualidade, formato de composição, resolução efetiva e demais presets necessários
- Tornar explícito quando o PDF final ficou dentro ou fora do orçamento

## Fora de escopo

- Alterar o output canônico de imagem do pipeline
- Garantir preservação perfeita de fidelidade visual em detrimento do orçamento de tamanho
- Ocultar um PDF superdimensionado como se fosse sucesso normal

## Estrutura de saída esperada

```text
runs/<run_id>/
  pdf/
    source/
      original.pdf
    extracted_pages/
      page-0001.png
      page-0002.png
    compose_ready_pages/
      page-0001.jpg
      page-0002.jpg
    rebuilt/
      original.upscaled.pdf
```

## Implementação esperada

1. Introduzir uma etapa específica de preparação para composição de PDF, separada da otimização genérica do passo 19.
2. Calcular o orçamento padrão do PDF final como:
   `max_pdf_size_bytes = pdf_original_size_bytes * 2`.
3. Permitir configuração explícita desse orçamento por CLI ou YAML, obedecendo `CLI > YAML > defaults`.
4. Gerar páginas em `pdf/compose_ready_pages/` a partir das imagens já com upscale, aplicando presets mais agressivos que os usados para compartilhamento de imagens isoladas quando necessário.
5. Implementar estratégia progressiva de compressão:
   tentar primeiro preservando mais qualidade e, se o orçamento não fechar, reduzir parâmetros de forma controlada até atingir o alvo ou esgotar a escada de presets.
6. Usar as páginas de `compose_ready_pages/` como fonte padrão para `pdf/rebuilt/`.
7. Persistir no manifesto e nos logs:
   tamanho do PDF original, orçamento alvo, tamanho estimado/final, razão final e preset efetivamente utilizado.
8. Se o alvo de tamanho não puder ser atingido, não tratar o PDF reconstruído como sucesso silencioso:
   a etapa deve registrar falha explícita ou status degradado claramente identificável.

## Sequência sugerida para implementação com Claude Code CLI

1. Ler os passos 19, 20 e 22 antes de editar código.
2. Mapear onde a composição do PDF escolhe as páginas fonte e inserir a nova camada `compose_ready_pages/`.
3. Definir o contrato de orçamento de tamanho e os presets progressivos.
4. Implementar compressão iterativa por página antes da recomposição.
5. Conectar a composição final ao novo conjunto de páginas preparadas.
6. Atualizar manifesto, logs e resumo final da CLI.
7. Cobrir testes com PDFs pequenos e um cenário representativo de orçamento restritivo.

## Critérios de aceite

- O PDF reconstruído passa a usar páginas preparadas especificamente para composição final.
- O sistema tenta por padrão manter o arquivo final em no máximo `2x` o tamanho do PDF original.
- Para um PDF original de `28 MB`, o alvo operacional padrão passa a ser `<= 56 MB`.
- Se o orçamento não for alcançável, o sistema deixa isso explícito e não mascara o resultado como sucesso normal.
- Os `outputs/*.png` canônicos continuam intactos.

## Como testar

- Executar o fluxo PDF com um documento pequeno e confirmar geração de `pdf/compose_ready_pages/`.
- Validar que o PDF final usa as páginas preparadas e não os PNGs canônicos brutos.
- Executar cenário com orçamento padrão e confirmar razão final `<= 2.0` quando atingível.
- Simular cenário em que o orçamento não pode ser alcançado e confirmar status explícito de falha ou degradação.
- Confirmar que o fluxo de imagem e o PDF sem essa nova etapa ainda permanecem auditáveis.

## Armadilhas a evitar

- Recompor o PDF final diretamente a partir dos PNGs de upscale sem preparação adicional.
- Tentar resolver o problema apenas comprimindo o PDF depois de pronto, sem controlar as páginas fonte.
- Tornar o orçamento de tamanho implícito ou invisível para o operador.
- Considerar qualquer arquivo final gerado como sucesso, mesmo quando ele continua impraticável.
