# Passo 25: Fase 2 — Pipeline Assíncrono com Prefetch de I/O

## Objetivo

Eliminar a ociosidade da GPU durante operações de disco implementando um pipeline
produtor-consumidor com três estágios paralelos: leitura antecipada (prefetch),
inferência GPU e escrita assíncrona. Ganho esperado: **1.3–1.8× adicional** sobre o
throughput obtido na Fase 1.

## ADRs relacionadas

- [0005-pipeline-deterministico-e-estrategia-de-erros.md](../adr/0005-pipeline-deterministico-e-estrategia-de-erros.md) — estratégia de erro por item preservada; determinismo de ordem mantido
- [0013-pipeline-assincrono-prefetch.md](../adr/0013-pipeline-assincrono-prefetch.md) — decisão de usar produtor-consumidor com `ThreadPoolExecutor`
- [0003-configuracao-com-precedencia-cli-yaml-defaults.md](../adr/0003-configuracao-com-precedencia-cli-yaml-defaults.md) — novos parâmetros seguem precedência `CLI > YAML > defaults`

## Problema a resolver

O pipeline serial atual mantém a GPU ociosa durante I/O de disco:

```
Estado atual (por imagem):
  [ler disco] → [CPU→GPU] → [forward pass] → [GPU→CPU] → [escrever disco]
       ↑                                                        ↑
       └─── GPU ociosa durante I/O ────────────────────────────┘
```

Estado alvo com pipeline assíncrono:

```
Thread de Leitura (pool)          Thread Principal          Thread de Escrita (pool)
        │                               │                           │
   task_1 →──────────── input_queue ──→ GPU forward ──→ output_queue ──→ disco
   task_2 →──── (prefetch)             │                           │
   task_3 →──── (prefetch)         próxima imagem          escreve em paralelo
```

## Dependências

- Passo 24 concluído (Fase 1 aplicada — modelo compilado, AMP, tile_pad correto)
- `concurrent.futures.ThreadPoolExecutor` e `queue.Queue` são stdlib Python (sem novas dependências)

## Entregáveis

1. Novo módulo `src/upscale_image/pipeline/async_worker.py` com `run_batch_async()`
2. `pipeline/batch.py` atualizado: parâmetro opcional `async_io`, `prefetch_size`, `write_workers`
3. `config/schema.py` atualizado: novos campos `async_io`, `prefetch_size`, `write_workers` em `RuntimeConfig`
4. `cli/main.py` atualizado: flags `--async-io` e `--prefetch`
5. `tests/test_async_worker.py` com cobertura da lógica assíncrona

## Escopo obrigatório

### 5.1 Novo módulo `async_worker.py`

**Arquivo a criar**: `src/upscale_image/pipeline/async_worker.py`

O módulo encapsula toda a lógica de I/O assíncrono. O pipeline principal (`batch.py`)
continua com uma interface idêntica à atual — as threads não vazam para fora deste módulo.

Função pública esperada:

```python
def run_batch_async(
    config: AppConfig,
    tasks: list[ImageTask],
    model: SuperResolutionModel,
    logger: RunLogger,
    prefetch_size: int = 4,
    write_workers: int = 2,
) -> list[ItemResult]:
```

Retorna `list[ItemResult]` **na mesma ordem** que `tasks`. A ordem de enfileiramento é
determinística (herda a ordem de `discover_images()`); apenas o I/O ocorre em paralelo.

**Estrutura interna obrigatória**:

- `_ReadResult`: dataclass com `task`, `image`, `error`, `t_read_start`
- `_InferResult`: dataclass com `task`, `output`, `inference_time_ms`, `input_w`, `input_h`, `error`, `t_read_start`
- Sentinel `_STOP = object()` para sinalizar fim de fila
- Thread de leitura: percorre `tasks` em ordem, coloca `_ReadResult` em `read_queue`
- Thread de inferência (thread principal): consome `read_queue`, chama `model.upscale()`, coloca `_InferResult` em `write_queue`
- Pool de escrita (`write_workers` threads): consome `write_queue`, chama `cv2.imwrite()`, registra em `logger`

**Tratamento de erros**: exceções em qualquer estágio são capturadas e convertidas em
`ItemResult(status="failed")` — conforme ADR 0005. Nenhuma exceção propaga para fora de
`run_batch_async()` por falhas de item individual.

**Sentinel e shutdown**: ao terminar, a thread de inferência coloca `write_workers`
cópias do sentinel `_STOP` em `write_queue`, para que cada writer thread receba exatamente um.

**Preservação de ordem**: indexar resultados por `tasks.index(item.task)` e retornar
`[results[i] for i in range(len(tasks))]` ao final.

### 5.2 Integração em `batch.py`

**Arquivo**: `src/upscale_image/pipeline/batch.py`

Adicionar parâmetros opcionais à função `run_batch()`:

```python
def run_batch(
    config: AppConfig,
    ctx: RunContext,
    model: SuperResolutionModel,
    logger: RunLogger,
    async_io: bool = False,
    prefetch_size: int = 4,
    write_workers: int = 2,
) -> BatchResult:
    ...
    if async_io:
        from upscale_image.pipeline.async_worker import run_batch_async
        results = run_batch_async(
            config, discovery.tasks, model, logger,
            prefetch_size=prefetch_size,
            write_workers=write_workers,
        )
    else:
        results = [_process_task(task, model, config, logger)
                   for task in discovery.tasks]
```

O comportamento serial padrão (`async_io=False`) deve permanecer **identicamente igual**
ao atual — sem nenhuma regressão.

### 5.3 Novos campos em `RuntimeConfig`

**Arquivo**: `src/upscale_image/config/schema.py`

Adicionar em `RuntimeConfig`:

```python
async_io: bool = False
prefetch_size: int = 4
write_workers: int = 2
```

### 5.4 Flags na CLI

**Arquivo**: `src/upscale_image/cli/main.py`

Adicionar ao comando `upscale`:

```python
async_io: bool = typer.Option(False, "--async-io",
    help="Overlap disk I/O and GPU inference (producer-consumer pipeline)")
prefetch: int = typer.Option(4, "--prefetch",
    help="Number of images to prefetch from disk")
```

Repassar para `run_batch()` via config ou diretamente.

## Fora de escopo

- Batch inference real (batch_size > 1) — é o Passo 26
- Qualquer alteração na interface `SuperResolutionModel`
- Mudanças na estrutura do run ou no manifesto

## Novos testes a criar

**Arquivo**: `tests/test_async_worker.py`

Cobrir:
- Resultado retornado na mesma ordem que `tasks` (critério fundamental)
- Falha de leitura em um item não interrompe os demais (ADR 0005)
- Falha de inferência em um item não interrompe os demais (ADR 0005)
- Falha de escrita em um item não interrompe os demais (ADR 0005)
- Shutdown limpo: nenhuma thread fica travada após conclusão do batch
- `run_batch()` com `async_io=True` produz resultados equivalentes ao modo serial (mesma lista de status)
- `run_batch()` com `async_io=False` (default) não sofre regressão

Usar `MockSuperResolutionModel` e imagens sintéticas em `tmp_path`.

## Sequência sugerida para implementação

1. Ler `src/upscale_image/pipeline/batch.py` e `src/upscale_image/pipeline/run.py` antes de editar.
2. Criar `async_worker.py` com a estrutura de dataclasses e o sentinel.
3. Implementar thread de leitura, thread de inferência e pool de escrita.
4. Integrar em `batch.py` com o parâmetro `async_io`.
5. Atualizar `config/schema.py` e `cli/main.py`.
6. Criar `tests/test_async_worker.py` e executar.
7. Executar `pytest tests/` completo para validar regressão.

## Critérios de aceite

- `run_batch_async()` retorna resultados na mesma ordem de `tasks`
- Falhas individuais em qualquer estágio geram `ItemResult(status="failed")` sem derrubar a run
- `run_batch()` com `async_io=False` é identicamente igual ao comportamento pré-existente
- `config.runtime.async_io`, `prefetch_size` e `write_workers` estão presentes em `RuntimeConfig`
- Flags `--async-io` e `--prefetch` estão disponíveis no comando `upscale`
- `pytest tests/` passa completamente

## Como testar

```bash
pytest tests/test_async_worker.py -v
pytest tests/test_batch.py -v      # regressão do modo serial
pytest tests/ -q                   # regressão completa
```

## Armadilhas a evitar

- Não usar `tasks.index()` para indexar resultados se houver tarefas duplicadas — usar `enumerate` com mapa `idx → result` ao invés
- Não colocar apenas um `_STOP` na `write_queue` com múltiplos write workers — cada worker precisa do seu próprio sentinel
- Não propagar exceções de item para fora de `run_batch_async()` — capturar tudo e converter em `ItemResult(status="failed")`
- Não alterar o comportamento do caminho `async_io=False` — o pipeline serial não deve ser tocado
