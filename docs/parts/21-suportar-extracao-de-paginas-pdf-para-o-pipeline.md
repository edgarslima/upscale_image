# Passo 21: Suportar Extração de Páginas PDF para o Pipeline

## Objetivo

Permitir que a aplicação trate um PDF como origem de páginas-imagem intermediárias, preparadas para o mesmo pipeline de `upscale` já usado para imagens.

## ADRs relacionadas

- [0002-persistencia-orientada-a-filesystem-e-runs.md](/home/edgar/dev/upscale_image/docs/adr/0002-persistencia-orientada-a-filesystem-e-runs.md)
- [0003-configuracao-com-precedencia-cli-yaml-defaults.md](/home/edgar/dev/upscale_image/docs/adr/0003-configuracao-com-precedencia-cli-yaml-defaults.md)
- [0005-pipeline-deterministico-e-estrategia-de-erros.md](/home/edgar/dev/upscale_image/docs/adr/0005-pipeline-deterministico-e-estrategia-de-erros.md)
- [0006-observabilidade-com-logs-manifesto-e-metricas.md](/home/edgar/dev/upscale_image/docs/adr/0006-observabilidade-com-logs-manifesto-e-metricas.md)

## Problema a resolver

O pipeline atual assume imagens em disco como entrada. Para suportar PDFs sem reescrever a lógica de inferência, é necessário transformar cada página do PDF em uma imagem intermediária auditável e determinística, preservando a relação entre documento, páginas e artefatos da `run`.

## Dependências

- Passo 03 concluído
- Passo 04 concluído
- Passo 09 concluído

## Entregáveis

- Camada de entrada PDF separada da descoberta de imagens existente
- Extração de páginas PDF para imagens intermediárias na `run`
- Estruturas de metadados que mapeiem `pdf -> páginas -> arquivos extraídos`
- Logs e manifesto com informações sobre a origem PDF
- Testes para PDFs válidos, PDFs inválidos e ordenação de páginas

## Escopo obrigatório

- Aceitar PDF apenas quando o modo de entrada indicar explicitamente esse tipo
- Extrair páginas em ordem estável e com numeração determinística
- Persistir páginas em diretório dedicado dentro da `run`
- Gerar nomes de arquivo previsíveis como `page-0001.png`
- Preparar a lista de páginas para ser consumida pelo pipeline existente como se fossem imagens comuns
- Registrar contagem total de páginas e eventuais falhas estruturais de extração

## Fora de escopo

- Reconstruir o PDF final
- Alterar benchmark para consumir PDFs
- Misturar descoberta de imagens e descoberta de PDF em uma única heurística implícita

## Estrutura de saída esperada

```text
runs/<run_id>/
  pdf/
    source/
      original.pdf
    extracted_pages/
      page-0001.png
      page-0002.png
```

## Implementação esperada

1. Introduzir um fluxo de ingestão específico para PDF, acionado por parâmetro explícito na CLI ou configuração efetiva.
2. Validar cedo a existência do arquivo PDF, sua legibilidade e a capacidade de extrair páginas antes de iniciar o processamento principal.
3. Persistir uma cópia ou referência controlada do PDF de origem dentro da `run`, conforme a ADR 0002.
4. Extrair cada página para uma imagem intermediária estável, em ordem natural de página.
5. Gerar uma estrutura de tarefas compatível com o pipeline já existente, reutilizando o máximo possível da camada de batch.
6. Registrar no manifesto informações mínimas:
   arquivo de origem, total de páginas, diretório de páginas extraídas e modo de entrada `pdf`.
7. Garantir que falhas estruturais de leitura do PDF abortem cedo, enquanto inconsistências detectadas depois da extração fiquem claramente registradas.

## Sequência sugerida para implementação com Claude Code CLI

1. Mapear o ponto atual onde a CLI assume diretório de imagens e separar a resolução de origem por tipo.
2. Criar o módulo de extração PDF desacoplado do pipeline de inferência.
3. Persistir páginas extraídas dentro da `run`.
4. Adaptar a preparação das tarefas para aceitar páginas extraídas.
5. Atualizar logs e manifesto.
6. Cobrir testes unitários e de integração da extração.

## Critérios de aceite

- Um PDF válido pode ser transformado em uma sequência determinística de páginas-imagem dentro da `run`.
- O pipeline consegue receber essas páginas sem reestruturação do núcleo de inferência.
- O modo atual de imagens continua funcionando sem alteração de comportamento.
- PDFs inválidos falham cedo com erro claro.

## Como testar

- Executar a extração sobre um PDF pequeno com múltiplas páginas.
- Confirmar nomes estáveis e ordenação correta de `page-0001`, `page-0002`, etc.
- Validar erro claro para PDF inexistente ou corrompido.
- Confirmar que o fluxo de imagens comum continua intacto.

## Armadilhas a evitar

- Inferir automaticamente que qualquer arquivo é PDF sem parâmetro explícito.
- Extrair páginas apenas em memória sem persistência auditável na `run`.
- Acoplar extração PDF diretamente ao runner do modelo.
- Perder o mapeamento entre página original e imagem intermediária.
