# Passo 29: Fase 5 — Multi-GPU com Worker Pool

## Objetivo

Distribuir o processamento de imagens entre múltiplas GPUs NVIDIA usando um pool de
processos independentes, obtendo ganho linear próximo a **N× por GPU adicionada**.
Cada GPU recebe um processo dedicado com sua própria cópia do modelo em memória — sem
compartilhamento de estado entre processos.

## ADRs relacionadas

- [0005-pipeline-deterministico-e-estrategia-de-erros.md](../adr/0005-pipeline-deterministico-e-estrategia-de-erros.md) — determinismo de ordem preservado; falhas por item são isoladas por processo
- [0013-pipeline-assincrono-prefetch.md](../adr/0013-pipeline-assincrono-prefetch.md) — multi-GPU é uma extensão natural do modelo de execução paralela; usa `multiprocessing` em vez de threads
- [0001-arquitetura-cli-local-monolitica.md](../adr/0001-arquitetura-cli-local-monolitica.md) — execução permanece local; multi-GPU é transparente para o usuário

## Problema a resolver

Uma única GPU processa imagens sequencialmente (mesmo com Fases 1–3). Em hardware com
múltiplas GPUs, o throughput pode ser multiplicado N× distribuindo as tasks entre GPUs
em processos separados. Threads não funcionam para isso — o GIL do Python e o contexto
CUDA impedem uso real de múltiplas GPUs em threads. Processos (`multiprocessing`) são
necessários.

## Dependências

- Passo 25 concluído (pipeline assíncrono — entende os padrões de producer-consumer)
- `torch.cuda.device_count() >= 2` para testar em hardware real
- `multiprocessing` é stdlib Python (sem novas dependências)

## Entregáveis

1. `src/upscale_image/pipeline/multi_gpu.py` — worker pool multi-GPU
2. `src/upscale_image/pipeline/batch.py` atualizado com branch `multi_gpu`
3. `src/upscale_image/config/schema.py` atualizado: `multi_gpu: bool = False`, `gpu_ids: list[int] = field(default_factory=list)`
4. `src/upscale_image/cli/main.py` atualizado: flag `--multi-gpu`
5. `tests/test_multi_gpu.py` cobrindo a lógica de distribuição via mocks

## Escopo obrigatório

### 8.1 Módulo `multi_gpu.py`

**Arquivo a criar**: `src/upscale_image/pipeline/multi_gpu.py`

**Arquitetura**:

```
Processo Principal
├── Descoberta de imagens → lista de tasks
├── Distribui tasks em task_queue (multiprocessing.Queue)
└── Coleta resultados de result_queue

Processo GPU:0           Processo GPU:1           Processo GPU:N
├── Carrega modelo       ├── Carrega modelo       ├── Carrega modelo
│   (isolado na GPU)         (isolado na GPU)         (isolado na GPU)
├── Consome da queue     ├── Consome da queue     ├── Consome da queue
└── Produz resultados    └── Produz resultados    └── Produz resultados
```

**Função de worker** (`_gpu_worker`):
- Recebe `gpu_id`, `model_factory`, `task_queue`, `result_queue`, `config`
- Define `os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)` antes de instanciar o modelo
- Consome tasks até receber `None` (poison pill) — then `model.unload()` e encerra
- Captura exceções por item e coloca `ItemResult(status="failed")` na `result_queue`

**Função pública** (`run_batch_multi_gpu`):

```python
def run_batch_multi_gpu(
    config: AppConfig,
    tasks: list[ImageTask],
    model_factory: Callable[[], SuperResolutionModel],
    gpu_ids: list[int] | None = None,
) -> list[ItemResult]:
```

- `gpu_ids=None` → detecta automaticamente via `torch.cuda.device_count()`
- Usa `mp.get_context("spawn")` para isolamento CUDA entre processos
- Enfileira todas as tasks, depois coloca N poison pills (uma por worker)
- Coleta todos os resultados de `result_queue`
- Retorna lista **na mesma ordem** de `tasks` (ordenar por `task.input_path` usando mapa)

**Nota sobre ordem**: workers podem processar em ordem diferente da enfileirada.
Usar `dict[str, ItemResult]` indexado por `task.input_path` e reconstruir a ordem
original ao final.

### 8.2 Integração em `batch.py`

**Arquivo**: `src/upscale_image/pipeline/batch.py`

Adicionar branch para multi-GPU:

```python
if config.runtime.multi_gpu and torch.cuda.device_count() > 1:
    from upscale_image.pipeline.multi_gpu import run_batch_multi_gpu
    gpu_ids = config.runtime.gpu_ids or None
    results = run_batch_multi_gpu(
        config, discovery.tasks, model_factory, gpu_ids=gpu_ids
    )
```

**Atenção**: `run_batch_multi_gpu` recebe `model_factory` (callable), não uma instância
de modelo já carregada. O modelo é carregado dentro de cada processo worker. A assinatura
de `run_batch()` pode precisar receber `model_factory` ao invés de `model` quando
`multi_gpu=True`, ou receber ambos e escolher o correto.

### 8.3 Novos campos em `RuntimeConfig`

**Arquivo**: `src/upscale_image/config/schema.py`

```python
from dataclasses import field

@dataclass
class RuntimeConfig:
    # ... campos existentes ...
    multi_gpu: bool      = False
    gpu_ids:   list[int] = field(default_factory=list)  # [] = usar todas as GPUs detectadas
```

### 8.4 Flag na CLI

**Arquivo**: `src/upscale_image/cli/main.py`

```python
multi_gpu: bool = typer.Option(False, "--multi-gpu",
    help="Distribute tasks across all available CUDA GPUs (requires 2+ GPUs)")
```

Validação: se `--multi-gpu` é passado mas `torch.cuda.device_count() < 2`, emitir
aviso claro e continuar em modo single-GPU (não abortar a run).

## Fora de escopo

- Ray-based scheduling (stdlib `multiprocessing` é suficiente para o caso de uso)
- Multi-GPU com ONNX Runtime ou TensorRT (requer integração adicional com aqueles runners)
- Balanceamento de carga dinâmico (distribuição estática por round-robin é suficiente)

## Novos testes a criar

**Arquivo**: `tests/test_multi_gpu.py`

Cobrir (todos via mocks — não requer hardware real):
- `run_batch_multi_gpu()` retorna resultados na mesma ordem que `tasks`
- Poison pills: cada worker recebe exatamente um `None`
- Falha em um item não interrompe os demais (mock `model.upscale` que levanta exceção para um item)
- `gpu_ids=None` com mock `torch.cuda.device_count() = 2` detecta 2 GPUs
- `run_batch()` com `multi_gpu=False` (default) não sofre regressão

**Estratégia de mock para CUDA**:

Usar `unittest.mock.patch("torch.cuda.device_count", return_value=2)` e
`unittest.mock.patch("torch.cuda.is_available", return_value=True)`.
Substituir a chamada de processo real por mock que retorna resultados fake
diretamente na `result_queue` (evitar `spawn` em testes unitários).

## Sequência sugerida para implementação

1. Ler `src/upscale_image/pipeline/batch.py` e entender o fluxo atual de `run_batch()`.
2. Criar `src/upscale_image/pipeline/multi_gpu.py` com `_gpu_worker` e `run_batch_multi_gpu`.
3. Atualizar `config/schema.py` com `multi_gpu` e `gpu_ids`.
4. Atualizar `pipeline/batch.py` com branch `multi_gpu`.
5. Atualizar `cli/main.py` com `--multi-gpu` e validação.
6. Criar `tests/test_multi_gpu.py` com mocks de CUDA e processo.
7. Executar `pytest tests/test_multi_gpu.py -v`.
8. Executar `pytest tests/` para regressão completa.

## Critérios de aceite

- `run_batch_multi_gpu()` existe e retorna `list[ItemResult]` na mesma ordem de `tasks`
- `_gpu_worker` usa `CUDA_VISIBLE_DEVICES` para isolamento por GPU
- `mp.get_context("spawn")` é usado (não `fork`) para compatibilidade com CUDA
- `config.runtime.multi_gpu` e `gpu_ids` existem em `RuntimeConfig`
- Flag `--multi-gpu` disponível na CLI com validação e aviso para < 2 GPUs
- `pytest tests/` passa completamente

## Como testar

```bash
pytest tests/test_multi_gpu.py -v
pytest tests/ -q   # regressão completa
```

## Armadilhas a evitar

- Não usar `mp.get_context("fork")` — CUDA não é seguro com fork; usar `"spawn"`
- Não carregar o modelo no processo principal e passar para workers — `multiprocessing` não serializa bem modelos PyTorch grandes
- Não usar `tasks.index(result.task)` para reordenar se tarefas puderem ter `input_path` repetido — usar mapa explícito por path
- Não ignorar o retorno de `result_queue.get()` sem checar se já coletou todos os resultados esperados
- Não abortar a run quando `--multi-gpu` é passado em hardware com < 2 GPUs — degradar graciosamente para single-GPU
