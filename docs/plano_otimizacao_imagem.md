# Plano de Otimização de Imagens

## Objetivo

Adicionar uma etapa de otimização de imagens já geradas pela aplicação sem comprometer a rastreabilidade, a reprodutibilidade e a comparabilidade das `runs`.

## Problema que o plano resolve

Hoje a aplicação gera `outputs/*.png` como artefato canônico da inferência. Isso é adequado para benchmark e auditoria técnica, mas não é o formato mais eficiente para distribuição, publicação ou armazenamento de longo prazo quando o objetivo principal passa a ser reduzir tamanho em disco e custo de transferência.

O plano de otimização precisa preservar três propriedades já consolidadas no projeto:

- o PNG gerado pela inferência continua sendo a fonte de verdade da `run`;
- a otimização não pode reescrever artefatos já usados por benchmark, comparação e relatório;
- toda derivação precisa ser rastreável por configuração, log e manifesto.

## Restrições arquiteturais

Este plano deve respeitar as ADRs já aceitas:

- `0002`: a `run` continua orientada a filesystem e todos os artefatos ficam encapsulados nela;
- `0003`: qualquer preset ou parâmetro de otimização precisa obedecer `CLI > YAML > defaults`;
- `0006`: a otimização deve produzir artefatos observáveis e auditáveis;
- `0009`: os dados produzidos precisam ser consumíveis por comparação e relatório;
- `0010`: a otimização é derivada e não destrutiva sobre os outputs canônicos.

## Incrementos escolhidos

Os incrementos de otimização serão implementados como passos adicionais do plano principal:

- `Passo 19`: gerar variantes otimizadas a partir dos PNGs canônicos da `run`, persistindo resultados em uma árvore separada e registrando economia por arquivo e por execução.
- `Passo 20`: integrar a otimização ao fluxo de `upscale` por parâmetro opcional, executando a etapa derivada automaticamente ao final da inferência quando solicitado.
- `Passo 21`: suportar entrada PDF, extraindo cada página em imagem intermediária para processamento determinístico no pipeline.
- `Passo 22`: recompor PDFs de saída a partir das páginas processadas e integrar a origem PDF ao comando `upscale` por parâmetro explícito.
- `Passo 23`: otimizar agressivamente as páginas antes da recomposição do PDF final, impondo orçamento de tamanho para que o PDF reconstruído permaneça compartilhável.

## Escopo do passo 19

- ler `runs/<run_id>/outputs/*.png` já produzidos;
- gerar derivados otimizados em formatos de distribuição, com presets determinísticos;
- persistir artefatos em `runs/<run_id>/optimized/`;
- salvar resumo estruturado por imagem e agregado;
- referenciar os artefatos de otimização no manifesto e nos logs da `run`.

## Escopo do passo 20

- adicionar ao fluxo de `upscale` um parâmetro opcional para habilitar otimização ao final da inferência;
- reutilizar a camada criada no passo 19, sem duplicar regras de serialização e persistência;
- manter a otimização desabilitada por padrão;
- executar a otimização somente após a `run` principal ter produzido seus `outputs/*.png`;
- preservar o mesmo comportamento determinístico, observável e não destrutivo do passo 19.

## Escopo do passo 21

- aceitar PDF como tipo de entrada apenas quando um parâmetro explícito indicar esse modo;
- extrair cada página do PDF para imagens intermediárias dentro da `run`;
- produzir uma lista determinística de páginas para o mesmo pipeline de upscale já existente;
- preservar mapeamento entre número de página, arquivo intermediário e artefatos finais da `run`;
- manter os fluxos atuais de imagem inalterados quando a origem não for PDF.

## Escopo do passo 22

- reconstruir um ou mais PDFs de saída a partir das páginas processadas e otimizadas;
- encadear extração, upscale, otimização opcional e recomposição quando a origem for PDF;
- manter o comando atual compatível com imagens individuais e diretórios de imagem;
- registrar no manifesto e nos logs que a `run` teve origem em PDF e quais PDFs derivados foram gerados.

## Escopo do passo 23

- otimizar as páginas processadas antes da recomposição do PDF em `pdf/rebuilt/`;
- usar um orçamento explícito de tamanho final para o PDF reconstruído;
- buscar como padrão que o PDF final não ultrapasse `2x` o tamanho do PDF original;
- aplicar compressão progressiva por página quando necessário, antes da etapa de composição final;
- falhar de forma explícita ou registrar status degradado quando o alvo de tamanho não puder ser atingido.

## Estrutura alvo

```text
runs/<run_id>/
  outputs/
    *.png
  optimized/
    summary.json
    per_image.csv
    webp/
      *.webp
    jpeg/
      *.jpg
```

## Estrutura alvo para origem PDF

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
  outputs/
    page-0001.png
    page-0002.png
  optimized/
    ...
```

## Decisão sobre etapas adicionais

O passo 19 continua sendo o incremento mínimo viável para otimização derivada e auditável. O passo 20 integra essa capacidade ao fluxo principal de `upscale` por parâmetro opcional. Para suporte a PDF, a decomposição em três passos adicionais é intencional: o passo 21 isola ingestão e extração de páginas, o passo 22 cobre recomposição do PDF e integração final à CLI, e o passo 23 introduz a camada de compressão agressiva orientada a orçamento para impedir que o PDF reconstruído fique impraticável para abertura e compartilhamento.
