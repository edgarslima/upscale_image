# ADR 0012: Estratégia de Batch Inference

## Status

Aceita

## Contexto

O pipeline atual processa uma imagem por chamada de `model.upscale()`. GPUs modernas
possuem capacidade de executar múltiplas imagens em paralelo dentro de um único forward
pass (batch inference). Um RTX 3080 processa um batch de 4 imagens 1080p em
aproximadamente 1.4× o tempo de 1 imagem, entregando 2.8× throughput por chamada.

O contrato `SuperResolutionModel` (ADR 0004) define apenas `upscale(image, config)`,
sem suporte a batch. Alterar a assinatura quebraria todos os runners existentes.

## Decisão

Adicionar o método `upscale_batch(images, config)` ao contrato `SuperResolutionModel`
com uma implementação default que chama `upscale()` em loop, garantindo compatibilidade
retroativa com todos os runners existentes (mock, bicubic, realesrgan).

Runners que suportam batch real sobrescrevem `upscale_batch()` com uma implementação
que processa todas as imagens em um único forward pass, com padding para igualar
dimensões dentro do batch e corte posterior para restaurar os tamanhos originais.

O pipeline aceita `batch_size` como parâmetro de `RuntimeConfig` (default: 1).
Com `batch_size=0`, o pipeline estima automaticamente o batch_size máximo seguro
baseado na VRAM disponível e no tamanho da primeira imagem.

Imagens são agrupadas por tamanho similar antes de serem enfileiradas, minimizando
o desperdício de VRAM causado por padding excessivo entre imagens de tamanhos muito
distintos. A tolerância de variação de tamanho dentro de um grupo é configurável.

## Consequências

- O pipeline de batch precisa ser modificado para agrupar tarefas antes de processá-las,
  alterando o loop por item para um loop por grupo de itens.
- `ItemResult` continua sendo produzido por imagem individual, preservando observabilidade.
- O manifesto e as métricas não mudam: a unidade de resultado continua sendo uma imagem.
- Runners que não sobrescrevem `upscale_batch()` comportam-se identicamente ao
  comportamento anterior (compatibilidade retroativa total).
- O pipeline assíncrono (ADR 0013) e o batch inference são complementares: o prefetch
  de I/O opera no nível de imagens individuais, enquanto o batch inference opera no
  nível de grupos na thread de inferência.
- Em CPU, batch inference não oferece ganho (CPU não paraleliza da mesma forma);
  `batch_size` é tratado como 1 quando `device=cpu`.
