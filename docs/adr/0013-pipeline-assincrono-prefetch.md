# ADR 0013: Pipeline Assíncrono com Prefetch de I/O

## Status

Aceita

## Contexto

O pipeline atual (ADR 0005) é síncrono: leitura do disco, inferência na GPU e escrita no
disco ocorrem sequencialmente para cada imagem. Durante a leitura e escrita, a GPU fica
ociosa; durante a inferência, o disco fica ocioso. Em imagens de alta resolução com uma
GPU rápida, a GPU fica ociosa por 5–15% do tempo total, e esse percentual cresce com
discos rápidos (NVMe) e modelos leves.

## Decisão

Separar I/O de disco e inferência GPU em estágios paralelos usando um modelo
produtor-consumidor com `threading.Thread` e `queue.Queue` da stdlib Python.

A arquitetura tem três estágios:
1. **Thread de Leitura**: lê imagens do disco com antecedência e coloca em fila de entrada.
2. **Thread Principal de Inferência**: consome da fila de entrada, executa o forward pass
   GPU e coloca resultados em fila de saída.
3. **Pool de Threads de Escrita**: consome da fila de saída e persiste imagens no disco
   em paralelo.

O tamanho das filas (prefetch_size) é configurável para controlar o uso de memória RAM.

## Consequências

- O ADR 0005 estabelece determinismo de ordem. Esta ADR preserva o determinismo de
  **ordem dos resultados** (o `BatchResult` e manifesto listam os itens na mesma ordem
  estável de `discover_images()`), mas relaxa o determinismo de **sequência de execução**
  (I/O e inferência ocorrem em threads paralelas).
- A estratégia de erro por item do ADR 0005 é preservada: exceções em qualquer estágio
  (leitura, inferência, escrita) são capturadas e convertidas em `ItemResult` com
  `status="failed"`, sem derrubar a run.
- O pipeline assíncrono é opt-in via `RuntimeConfig.async_io = True`. O comportamento
  serial padrão é mantido para compatibilidade e para runs com poucas imagens onde o
  overhead de gerenciamento de threads não se justifica.
- O módulo `async_worker.py` encapsula toda a lógica de threads. O `batch.py` permanece
  com uma interface pública estável, sem expor threads ao chamador.
- A thread de inferência GPU é única (single-threaded), o que é correto: múltiplas threads
  concorrentes chamando a GPU não oferecem ganho e introduzem overhead de sincronização.
  O paralelismo real está no I/O, não na inferência.
- Memória RAM adicional: até `prefetch_size` imagens descomprimidas (tipicamente
  1080p RGB float32 = ~25 MB por imagem) ficam em buffer simultâneo.
