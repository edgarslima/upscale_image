# Passo 22: Recompor PDF de Saída e Integrar Origem PDF ao Upscale

## Objetivo

Completar o fluxo de PDF no comando `upscale`, reconstruindo um novo PDF a partir das páginas processadas e otimizadas quando a origem da execução for um PDF.

## ADRs relacionadas

- [0002-persistencia-orientada-a-filesystem-e-runs.md](/home/edgar/dev/upscale_image/docs/adr/0002-persistencia-orientada-a-filesystem-e-runs.md)
- [0003-configuracao-com-precedencia-cli-yaml-defaults.md](/home/edgar/dev/upscale_image/docs/adr/0003-configuracao-com-precedencia-cli-yaml-defaults.md)
- [0005-pipeline-deterministico-e-estrategia-de-erros.md](/home/edgar/dev/upscale_image/docs/adr/0005-pipeline-deterministico-e-estrategia-de-erros.md)
- [0006-observabilidade-com-logs-manifesto-e-metricas.md](/home/edgar/dev/upscale_image/docs/adr/0006-observabilidade-com-logs-manifesto-e-metricas.md)
- [0010-otimizacao-derivada-e-nao-destrutiva-dos-outputs.md](/home/edgar/dev/upscale_image/docs/adr/0010-otimizacao-derivada-e-nao-destrutiva-dos-outputs.md)

## Problema a resolver

Depois de extrair páginas e processá-las como imagens, ainda falta devolver ao usuário um artefato PDF final equivalente ao original, porém com as páginas melhoradas. Além disso, essa cadeia completa precisa ser acionada por parâmetro no mesmo comando `upscale`, sem quebrar o modo atual de imagens.

## Dependências

- Passo 19 concluído
- Passo 20 concluído
- Passo 21 concluído

## Entregáveis

- Parâmetro explícito no comando `upscale` para indicar origem PDF
- Fluxo encadeado:
  extração de páginas -> upscale -> otimização opcional -> recomposição do PDF
- Diretório `pdf/rebuilt/` na `run`
- Manifesto e logs registrando PDFs gerados
- Testes cobrindo o caminho de imagem e o caminho de PDF

## Escopo obrigatório

- Adicionar parâmetro claro para indicar que a origem da entrada é PDF
- Manter o fluxo atual de imagens quando esse parâmetro não for enviado
- Reconstruir o PDF de saída preservando a ordem original das páginas
- Usar as páginas processadas da `run` como fonte para recomposição
- Se a otimização estiver habilitada, definir de forma explícita quais artefatos alimentam a recomposição do PDF final
- Persistir o PDF final em diretório dedicado dentro da `run`

## Fora de escopo

- OCR
- Preservação completa de elementos vetoriais internos do PDF original
- Benchmark direto em nível de PDF
- Misturar saída PDF reconstruída com `outputs/` de imagem

## Estrutura de saída esperada

```text
runs/<run_id>/
  pdf/
    source/
      original.pdf
    extracted_pages/
      page-0001.png
      page-0002.png
    rebuilt/
      original.upscaled.pdf
```

## Implementação esperada

1. Estender a CLI de `upscale` com um parâmetro explícito de tipo de entrada ou modo PDF.
2. Resolver a origem da entrada sem ambiguidade:
   se `input_mode=image`, manter o comportamento atual; se `input_mode=pdf`, ativar o fluxo encadeado de PDF.
3. Reutilizar a extração do passo 21 e a otimização dos passos 19 e 20, sem duplicar lógica.
4. Definir uma política explícita para recomposição:
   por padrão, o PDF final deve ser montado a partir das páginas melhoradas canônicas; se a otimização estiver habilitada e for compatível com a reconstrução, registrar claramente a origem dos artefatos usados.
5. Persistir o PDF reconstruído em `pdf/rebuilt/`.
6. Atualizar o manifesto com:
   modo de entrada, PDF de origem, número de páginas, PDF final gerado e relação com artefatos intermediários.
7. Garantir que falhas na recomposição do PDF não corrompam os outputs já gerados, mas registrem claramente o estado final da etapa derivada.

## Sequência sugerida para implementação com Claude Code CLI

1. Ler os passos 19, 20 e 21 antes de editar código.
2. Definir o novo parâmetro da CLI para origem PDF.
3. Encadear a resolução da entrada PDF ao pipeline já existente.
4. Implementar a reconstrução do PDF a partir das páginas processadas.
5. Atualizar manifesto, logs e resumo final da execução.
6. Cobrir testes do fluxo completo de PDF e do caminho tradicional de imagens.

## Critérios de aceite

- O comando `upscale` continua aceitando imagens como hoje quando o modo PDF não é usado.
- Quando o modo PDF é informado, a aplicação extrai páginas, executa upscale, aplica otimização opcional e gera um novo PDF.
- A ordem das páginas do PDF reconstruído corresponde à ordem do PDF original.
- Os artefatos intermediários e finais ficam auditáveis dentro da `run`.
- O manifesto deixa explícito que a execução teve origem em PDF.

## Como testar

- Executar `upscale` em modo imagem e confirmar que nada muda.
- Executar `upscale` em modo PDF com um documento pequeno e validar extração, outputs por página e PDF final.
- Executar `upscale` em modo PDF com otimização habilitada e validar coexistência entre `outputs/`, `optimized/` e `pdf/rebuilt/`.
- Simular falha na recomposição do PDF e confirmar que a run continua auditável.

## Armadilhas a evitar

- Inserir heurística implícita de tipo de entrada sem parâmetro explícito.
- Reescrever a lógica do pipeline de imagens dentro do fluxo PDF.
- Perder a ordem das páginas ao reconstruir o PDF.
- Esconder no PDF final a origem dos artefatos usados.
