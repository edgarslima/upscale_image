# Plano de Otimização de Performance — upscale_image

**Data**: 2026-03-29
**Baseado em**: `docs/analise_escala.md`
**Linguagem alvo**: Python (sem reescrita)
**Horizonte**: 4 fases incrementais, implementáveis independentemente

---

## 1. Diagnóstico Resumido

O código atual é serial e não utiliza os recursos disponíveis do hardware:

```
Estado atual (por imagem):
  [ler disco] → [CPU→GPU] → [forward pass] → [GPU→CPU] → [escrever disco]
       ↑                                                        ↑
       └─── GPU ociosa durante I/O ────────────────────────────┘
```

Quatro problemas centrais identificados em `pipeline/batch.py` e `models/realesrgan.py`:

| Problema | Local no código | Impacto |
|---|---|---|
| Execução serial imagem por imagem | `batch.py:154` — loop `for task in discovery.tasks` | Fundacional |
| Modelo não compilado | `realesrgan.py:115` — `net.eval()`, sem `torch.compile` | 20–40% |
| AMP manual e incompleto | `realesrgan.py:154` — `tensor.half()` manual, sem `autocast` | 30–50% em CUDA |
| `tile_pad` muito pequeno | `schema.py:23` — default `tile_pad=10` | Artefatos visuais |

---

## 2. Dependências Novas

### 2.1 Por Fase

#### Fase 1 (Quick Wins) — sem novas dependências
Usa apenas o que já está instalado. Todas as APIs (`torch.compile`, `torch.autocast`, `torch.backends.cudnn`) estão disponíveis no `torch==2.11.0` presente em `requirements/base.txt`.

#### Fase 2 (Pipeline Assíncrono) — sem novas dependências externas
`concurrent.futures.ThreadPoolExecutor` e `queue.Queue` são stdlib Python. `torch.utils.data.DataLoader` está em `torch` (já instalado).

#### Fase 3 (Batch Inference Real) — sem novas dependências
Usa NumPy (já instalado via PyTorch) e stdlib.

#### Fase 4 (TensorRT / ONNX Runtime)

```
# requirements/performance.txt  ← arquivo NOVO a criar
-r base.txt

# Opção A: TensorRT via torch-tensorrt (recomendado — permanece no ecossistema PyTorch)
torch-tensorrt==2.11.0          # versão alinhada com torch==2.11.0
tensorrt>=10.0,<11.0            # instalado via pip ou via CUDA toolkit

# Opção B: ONNX Runtime (cross-vendor — AMD, CPU, etc.)
onnxruntime-gpu==1.20.0         # com CUDA Execution Provider
# OU, para TensorRT via ONNX Runtime:
# onnxruntime-tensorrt           # plugin separado, versões variam por TRT
```

> **Decisão de exclusão mútua**: torch-tensorrt e onnxruntime-gpu **não devem** ser instalados juntos no mesmo ambiente — conflitos de CUDA runtime são frequentes. Criar ambientes separados ou usar extras do pyproject.toml.

#### Fase 5 (Multi-GPU)

```
# Adicionar a requirements/performance.txt
ray[default]==2.40.0    # scheduler + worker pool multi-GPU
# OU, alternativa mais leve (sem Ray):
# stdlib multiprocessing é suficiente para caso simples
```

### 2.2 Arquivo `requirements/performance.txt` (novo)

```
# Performance tier: TensorRT + async workers
-r base.txt
torch-tensorrt==2.11.0
```

### 2.3 Alterações em `pyproject.toml`

Adicionar optional extras para não forçar dependências pesadas em todos os usuários:

```toml
[project.optional-dependencies]
performance = ["torch-tensorrt==2.11.0"]
multi-gpu   = ["ray[default]==2.40.0"]
dev         = ["pytest==9.0.2"]
benchmark   = ["scikit-image==0.26.0", "pyiqa==0.1.15.post2"]
```

---

## 3. ADRs Afetadas e Novas ADRs

### 3.1 ADRs Existentes que Precisam de Atualização

#### ADR 0005 — Pipeline Determinístico e Estratégia de Erros

**Situação atual**: "Executar o pipeline de forma **síncrona** e determinística".

**Conflito com Fase 2**: o pipeline assíncrono com prefetch introduz concorrência (threads de I/O rodando em paralelo com a inferência GPU). O determinismo de **ordem** é preservado; o determinismo de **sequência de execução** não é.

**Atualização necessária**: a ADR deve ser atualizada para distinguir:
- *Determinismo de ordem*: mantido — as tarefas são enfileiradas na mesma ordem estável produzida por `discover_images()`.
- *Determinismo de execução*: relaxado para permitir I/O em threads paralelas, desde que o resultado seja idêntico a uma execução serial (os resultados por item não dependem de outros itens).
- A estratégia de erros permanece igual: falha por item é isolada.

**Arquivo a editar**: `docs/adr/0005-pipeline-deterministico-e-estrategia-de-erros.md`

#### ADR 0007 — Stack Tecnológica Principal

**Situação atual**: lista fixa de bibliotecas sem menção a compilação de modelo ou aceleração de inferência.

**Atualização necessária**: adicionar às consequências:
- `torch.compile` e `torch.backends.cudnn.benchmark` como padrão para execução em CUDA.
- `torch.autocast` como mecanismo preferencial de precisão mista (substituindo a conversão manual `tensor.half()`).
- `torch-tensorrt` como dependência opcional para aceleração máxima (Fase 4).

**Arquivo a editar**: `docs/adr/0007-stack-tecnologica-principal.md`

### 3.2 Novas ADRs a Criar

#### ADR 0011 — Compilação de Modelo e AMP Automático

**Decisão**: aplicar `torch.compile()` e `torch.autocast` em todos os runners CUDA como comportamento padrão, ativado automaticamente quando `device=cuda`.

**Arquivo a criar**: `docs/adr/0011-compilacao-de-modelo-e-amp-automatico.md`

#### ADR 0012 — Estratégia de Batching na Inferência

**Decisão**: introduzir processamento em lote (batch_size > 1) como opção configurável, com agrupamento por dimensão e gerenciamento dinâmico de VRAM. O pipeline base continua suportando batch_size=1 para compatibilidade.

**Arquivo a criar**: `docs/adr/0012-batch-inference.md`

#### ADR 0013 — Pipeline Assíncrono com Prefetch de I/O

**Decisão**: separar I/O de disco e inferência GPU em estágios paralelos via produtor-consumidor com `ThreadPoolExecutor`, preservando a estratégia de erro por item do ADR 0005.

**Arquivo a criar**: `docs/adr/0013-pipeline-assincrono-prefetch.md`

#### ADR 0014 — Backend de Inferência Plugável (TensorRT / ONNX Runtime)

**Decisão**: abstrair o backend de execução do modelo (PyTorch eager, torch.compile, TensorRT) atrás da interface `SuperResolutionModel` existente (ADR 0004), sem alterar o contrato do pipeline.

**Arquivo a criar**: `docs/adr/0014-backend-de-inferencia-plugavel.md`

---

## 4. Fase 1 — Quick Wins

**Esforço**: 1–2 dias | **Ganho esperado**: 2–3× throughput | **Risco**: baixo

Todas as mudanças são aditivas e retrocompatíveis. Nenhuma interface pública muda.

### 4.1 Corrigir `tile_pad` default (artefatos visuais)

**Arquivo**: `src/upscale_image/config/schema.py`

**Problema**: `tile_pad=10` produz artefatos visíveis nas bordas de tiles. O padding precisa ser suficiente para cobrir o campo receptivo do modelo — no RRDBNet x4, o campo receptivo efetivo requer padding mínimo de 32.

**Mudança**:
```python
# Antes:
tile_pad: int = 10

# Depois:
tile_pad: int = 32
```

**Impacto em testes**: `tests/test_config.py` e qualquer teste que verifique o valor default de `tile_pad` precisam ser atualizados.

### 4.2 Habilitar `cudnn.benchmark` na inicialização do modelo

**Arquivo**: `src/upscale_image/models/realesrgan.py`

**Onde inserir**: no método `load()`, após `net.eval()`, antes de retornar.

**Lógica**:
```python
def load(self) -> None:
    # ... código existente de carregamento de pesos ...
    net.eval()

    # Habilitar cuDNN autotuner: seleciona o algoritmo de convolução mais
    # rápido para os tamanhos de tensor encontrados na primeira execução.
    # Custo: primeiras chamadas são ~2× mais lentas enquanto o autotuner roda.
    # Benefício: todas as chamadas subsequentes usam o algoritmo ótimo.
    # Condição: desabilitar para benchmarks que exijam determinismo absoluto.
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    self._net = net
    self._loaded = True
```

**Nota**: `cudnn.benchmark=True` é um estado global do processo. Se múltiplos runners forem instanciados (Fase 5), a flag já estará ativa.

### 4.3 Substituir AMP manual por `torch.autocast`

**Arquivo**: `src/upscale_image/models/realesrgan.py`

**Problema atual**: a conversão para FP16 é feita manualmente (`net.half()`, `tensor.half()`). Isso é frágil — qualquer operação que precise de FP32 internamente causa `nan` silencioso. O `torch.autocast` gerencia automaticamente quais ops usam FP16 (convoluções, matmuls) e quais permanecem em FP32 (softmax, somas de perda).

**Mudança no método `upscale()`**:
```python
def upscale(self, image: np.ndarray, config: AppConfig) -> np.ndarray:
    if not self._loaded or self._net is None:
        raise RuntimeError("Model not loaded. Call load() before upscale().")

    device = self._resolve_device(config.runtime.device)
    net = self._net.to(device)

    # Modelo sempre em FP32 em memória; autocast gerencia a precisão por op.
    # FP16 manual é substituído por autocast — mais seguro e igualmente rápido.
    net = net.float()

    # BGR uint8 → RGB float32 tensor [0, 1] — sempre FP32 na entrada
    img_rgb = image[:, :, ::-1].copy()
    tensor = (
        torch.from_numpy(img_rgb)
        .permute(2, 0, 1)
        .float()
        .div(255.0)
        .unsqueeze(0)
        .to(device)
    )

    use_amp = (config.runtime.precision == "fp16") and (device.type == "cuda")

    with torch.inference_mode():
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            if config.runtime.tile_size > 0:
                output = self._upscale_tiled(tensor, net, config)
            else:
                output = net(tensor)

    # Saída sempre volta para FP32 antes de converter para uint8
    out_np = (
        output.squeeze(0)
        .float()
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .byte()
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    return out_np[:, :, ::-1].copy()
```

### 4.4 Aplicar `torch.compile` no carregamento do modelo

**Arquivo**: `src/upscale_image/models/realesrgan.py`

**Onde inserir**: no método `load()`, após `net.eval()` e após configurar `cudnn.benchmark`.

```python
def load(self) -> None:
    # ... carregamento e validação de pesos ...
    net.eval()

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        # torch.compile aplica fusão de kernels e otimizações de grafo.
        # mode="reduce-overhead": menor latência por chamada (ideal para batch pequeno).
        # mode="max-autotune": maior throughput (ideal para batch grande, Fase 3).
        # A compilação ocorre na primeira chamada de forward (lazy), não aqui.
        net = torch.compile(net, mode="reduce-overhead")

    self._net = net
    self._loaded = True
```

**Atenção**: `torch.compile` transforma o modelo em um objeto `OptimizedModule`. Isso é compatível com a interface `nn.Module` — `.to(device)`, `.float()`, `.half()` funcionam normalmente. O tipo não é mais `RRDBNet` mas o contrato público não expõe isso.

**Impacto em testes**: `tests/test_realesrgan_runner.py` — testes que verificam `isinstance(model._net, RRDBNet)` precisam ser ajustados para `isinstance(model._net, torch._dynamo.OptimizedModule)` ou remover a verificação de tipo interno (preferível — é detalhe de implementação).

### 4.5 Aumentar `tile_pad` no tiling e implementar Gaussian feathering

**Arquivo**: `src/upscale_image/models/realesrgan.py` — método `_upscale_tiled`

**Problema atual**: os tiles são colados sem blend. Com `tile_pad=32`, o corte duro ainda pode produzir bordas se a imagem tiver gradientes. A solução definitiva é feathering gaussiano: aplicar uma máscara que suaviza a contribuição de cada tile nas bordas.

**Implementação**:
```python
def _upscale_tiled(
    self, tensor: torch.Tensor, net: nn.Module, config: "AppConfig"
) -> torch.Tensor:
    tile_size = config.runtime.tile_size
    tile_pad  = config.runtime.tile_pad
    scale     = self._scale

    _, channel, height, width = tensor.shape
    # Acumuladores: soma ponderada e soma de pesos
    output_sum     = tensor.new_zeros(1, channel, height * scale, width * scale)
    weight_sum     = tensor.new_zeros(1, 1,       height * scale, width * scale)

    tiles_x = math.ceil(width  / tile_size)
    tiles_y = math.ceil(height / tile_size)

    for iy in range(tiles_y):
        for ix in range(tiles_x):
            in_x1 = ix * tile_size
            in_x2 = min(in_x1 + tile_size, width)
            in_y1 = iy * tile_size
            in_y2 = min(in_y1 + tile_size, height)

            px1 = max(in_x1 - tile_pad, 0)
            px2 = min(in_x2 + tile_pad, width)
            py1 = max(in_y1 - tile_pad, 0)
            py2 = min(in_y2 + tile_pad, height)

            tile_in = tensor[:, :, py1:py2, px1:px2]
            with torch.inference_mode():
                tile_out = net(tile_in)

            # Região válida (sem padding) no espaço de saída
            cx1 = (in_x1 - px1) * scale
            cx2 = cx1 + (in_x2 - in_x1) * scale
            cy1 = (in_y1 - py1) * scale
            cy2 = cy1 + (in_y2 - in_y1) * scale

            valid = tile_out[:, :, cy1:cy2, cx1:cx2]

            # Máscara de blend: janela de Hann (suave nas bordas, 1 no centro)
            h_out = cy2 - cy1
            w_out = cx2 - cx1
            mask = self._hann_window(h_out, w_out, device=tensor.device)

            out_y1 = in_y1 * scale
            out_y2 = in_y2 * scale
            out_x1 = in_x1 * scale
            out_x2 = in_x2 * scale

            output_sum[:, :, out_y1:out_y2, out_x1:out_x2] += valid * mask
            weight_sum[:,  :, out_y1:out_y2, out_x1:out_x2] += mask

    # Normalizar pela soma de pesos (evita divisão por zero com clamp)
    return output_sum / weight_sum.clamp(min=1e-6)

@staticmethod
def _hann_window(h: int, w: int, device: torch.device) -> torch.Tensor:
    """Janela de Hann 2D: suave nas bordas, máximo 1 no centro."""
    hann_h = torch.hann_window(h, periodic=False, device=device)
    hann_w = torch.hann_window(w, periodic=False, device=device)
    # outer product → (H, W); reshape para (1, 1, H, W)
    return (hann_h.unsqueeze(1) * hann_w.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
```

---

## 5. Fase 2 — Pipeline Assíncrono com Prefetch

**Esforço**: 1 semana | **Ganho esperado**: 1.3–1.8× adicional sobre Fase 1 | **Risco**: médio

**Objetivo**: enquanto a GPU processa a imagem N, o disco lê a imagem N+1 e escreve a imagem N-1 simultaneamente. GPU nunca fica ociosa esperando I/O.

### 5.1 Arquitetura Producer-Consumer

```
Thread de Leitura (pool)          Thread Principal          Thread de Escrita (pool)
        │                               │                           │
   task_1 →──────────── input_queue ──→ GPU forward ──→ output_queue ──→ disco
   task_2 →──── (prefetch)             │                           │
   task_3 →──── (prefetch)         próxima imagem          escreve em paralelo
```

### 5.2 Novo módulo: `src/upscale_image/pipeline/async_worker.py`

Este módulo encapsula toda a lógica assíncrona, deixando `batch.py` com uma interface igual à atual (sem expor threads ao pipeline principal).

```python
"""Pipeline assíncrono: I/O de leitura e escrita em threads paralelas à inferência GPU.

Preserva a estratégia de erro por item do ADR 0005: exceções em qualquer fase
são capturadas e convertidas em ItemResult com status="failed".

A ordem de enfileiramento é determinística (mesma ordem de discover_images).
A ordem de conclusão pode diferir apenas nas threads de escrita, mas o
manifesto e o BatchResult preservam a ordem original de enfileiramento.
"""
from __future__ import annotations

import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from upscale_image.config import AppConfig
from upscale_image.io.task import ImageTask
from upscale_image.models.base import SuperResolutionModel
from upscale_image.pipeline.batch import ItemResult
from upscale_image.pipeline.logger import RunLogger

if TYPE_CHECKING:
    pass

# Sentinel para sinalizar fim de fila
_STOP = object()


@dataclass
class _ReadResult:
    task: ImageTask
    image: np.ndarray | None
    error: str | None
    t_read_start: float


@dataclass
class _InferResult:
    task: ImageTask
    output: np.ndarray | None
    inference_time_ms: float | None
    input_w: int | None
    input_h: int | None
    error: str | None
    t_read_start: float


def run_batch_async(
    config: AppConfig,
    tasks: list[ImageTask],
    model: SuperResolutionModel,
    logger: RunLogger,
    prefetch_size: int = 4,
    write_workers: int = 2,
) -> list[ItemResult]:
    """Executa inferência com I/O assíncrono em threads paralelas.

    Args:
        config:        Configuração resolvida.
        tasks:         Lista ordenada de ImageTask (produzida por discover_images).
        model:         Runner carregado.
        logger:        Logger da run.
        prefetch_size: Quantas imagens pré-carregar no buffer de entrada.
        write_workers: Threads de escrita paralela.

    Returns:
        Lista de ItemResult na mesma ordem que tasks.
    """
    read_queue: queue.Queue[_ReadResult | object]   = queue.Queue(maxsize=prefetch_size)
    write_queue: queue.Queue[_InferResult | object] = queue.Queue(maxsize=prefetch_size)

    # Mapa para preservar ordem: índice original → ItemResult
    results: dict[int, ItemResult] = {}
    results_lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # Thread de Leitura                                                    #
    # ------------------------------------------------------------------ #
    def reader():
        for task in tasks:
            t_start = time.monotonic()
            try:
                image = cv2.imread(task.input_path)
                if image is None:
                    raise RuntimeError(
                        f"cv2.imread returned None for {task.input_path!r}"
                    )
                read_queue.put(_ReadResult(task=task, image=image,
                                           error=None, t_read_start=t_start))
            except Exception as exc:  # noqa: BLE001
                read_queue.put(_ReadResult(task=task, image=None,
                                           error=str(exc), t_read_start=t_start))
        read_queue.put(_STOP)

    # ------------------------------------------------------------------ #
    # Thread Principal — Inferência GPU (single thread, sem GIL em CUDA) #
    # ------------------------------------------------------------------ #
    def infer_loop():
        while True:
            item = read_queue.get()
            if item is _STOP:
                write_queue.put(_STOP)
                break
            assert isinstance(item, _ReadResult)

            if item.error:
                write_queue.put(_InferResult(
                    task=item.task, output=None,
                    inference_time_ms=None,
                    input_w=None, input_h=None,
                    error=item.error,
                    t_read_start=item.t_read_start,
                ))
                continue

            try:
                h, w = item.image.shape[:2]
                t_infer = time.monotonic()
                output = model.upscale(item.image, config)
                infer_ms = (time.monotonic() - t_infer) * 1000.0

                write_queue.put(_InferResult(
                    task=item.task, output=output,
                    inference_time_ms=infer_ms,
                    input_w=w, input_h=h,
                    error=None,
                    t_read_start=item.t_read_start,
                ))
            except Exception as exc:  # noqa: BLE001
                write_queue.put(_InferResult(
                    task=item.task, output=None,
                    inference_time_ms=None,
                    input_w=None, input_h=None,
                    error=str(exc),
                    t_read_start=item.t_read_start,
                ))

    # ------------------------------------------------------------------ #
    # Pool de Escrita (write_workers threads)                             #
    # ------------------------------------------------------------------ #
    stop_count = 0
    stop_lock  = threading.Lock()

    def writer():
        nonlocal stop_count
        while True:
            item = write_queue.get()
            if item is _STOP:
                with stop_lock:
                    stop_count += 1
                    if stop_count >= write_workers:
                        pass  # todos terminaram
                break
            assert isinstance(item, _InferResult)

            elapsed = time.monotonic() - item.t_read_start

            if item.error:
                item.task.status = "failed"
                logger.log_item_error(item.task, RuntimeError(item.error))
                result = ItemResult(
                    task=item.task, status="failed",
                    elapsed=elapsed, error=item.error,
                )
            else:
                try:
                    Path(item.task.output_path).parent.mkdir(parents=True, exist_ok=True)
                    success = cv2.imwrite(item.task.output_path, item.output)
                    if not success:
                        raise RuntimeError(
                            f"cv2.imwrite failed for {item.task.output_path!r}"
                        )
                    item.task.status = "done"
                    logger.log_item_done(item.task, elapsed)
                    out_h, out_w = item.output.shape[:2]
                    result = ItemResult(
                        task=item.task, status="done",
                        elapsed=elapsed,
                        inference_time_ms=item.inference_time_ms,
                        input_width=item.input_w,
                        input_height=item.input_h,
                        output_width=out_w,
                        output_height=out_h,
                    )
                except Exception as exc:  # noqa: BLE001
                    item.task.status = "failed"
                    logger.log_item_error(item.task, exc)
                    result = ItemResult(
                        task=item.task, status="failed",
                        elapsed=elapsed, error=str(exc),
                    )

            # Indexar pelo índice original para preservar ordem
            idx = tasks.index(item.task)
            with results_lock:
                results[idx] = result

    # ------------------------------------------------------------------ #
    # Orquestração                                                        #
    # ------------------------------------------------------------------ #
    reader_thread = threading.Thread(target=reader, daemon=True, name="sr-reader")
    infer_thread  = threading.Thread(target=infer_loop, daemon=True, name="sr-infer")

    # O STOP do write_queue precisa ser copiado write_workers vezes
    # porque cada worker consome apenas o seu próprio STOP.
    # Solução: infer_loop coloca N STOPs.
    # Ajustar infer_loop para colocar write_workers STOPs:
    # (implementação inline acima simplificada para clareza — ver nota abaixo)

    with ThreadPoolExecutor(max_workers=write_workers, thread_name_prefix="sr-writer") as executor:
        reader_thread.start()
        infer_thread.start()
        write_futures = [executor.submit(writer) for _ in range(write_workers)]

        reader_thread.join()
        infer_thread.join()
        for f in write_futures:
            f.result()

    # Retornar na ordem original
    return [results[i] for i in range(len(tasks))]
```

> **Nota sobre o STOP sentinel**: na implementação real, `infer_loop` deve colocar `write_workers` cópias do sentinel `_STOP` na `write_queue` ao terminar, para que cada writer thread receba um. Omitido acima para clareza, mas obrigatório na implementação.

### 5.3 Integração com `batch.py`

`run_batch()` em `batch.py` recebe uma flag opcional `async_io`:

```python
def run_batch(
    config: AppConfig,
    ctx: RunContext,
    model: SuperResolutionModel,
    logger: RunLogger,
    async_io: bool = False,          # ← novo parâmetro
    prefetch_size: int = 4,          # ← novo parâmetro
    write_workers: int = 2,          # ← novo parâmetro
) -> BatchResult:
    discovery = discover_images(config.input_dir, str(ctx.outputs_dir))
    logger.log_run_start(...)
    logger.log_skipped_files(discovery.skipped)

    run_start = time.monotonic()

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

    total_elapsed = time.monotonic() - run_start
    batch = BatchResult(results=results, skipped=discovery.skipped,
                        total_elapsed_s=total_elapsed)
    logger.log_run_summary(...)
    return batch
```

### 5.4 Exposição via CLI e Config

**`config/schema.py`** — adicionar em `RuntimeConfig`:
```python
@dataclass
class RuntimeConfig:
    device: str = "cpu"
    precision: Literal["fp32", "fp16"] = "fp32"
    tile_size: int = 0
    tile_pad: int = 32          # atualizado de 10 → 32
    async_io: bool = False      # ← novo
    prefetch_size: int = 4      # ← novo
    write_workers: int = 2      # ← novo
    batch_size: int = 1         # ← preparação para Fase 3
```

**`cli/main.py`** — adicionar flags ao comando `upscale`:
```python
async_io: bool = typer.Option(False, "--async-io", help="Overlap I/O and GPU inference"),
prefetch: int  = typer.Option(4,     "--prefetch",  help="Images to prefetch from disk"),
```

---

## 6. Fase 3 — Batch Inference Real

**Esforço**: 1–2 semanas | **Ganho esperado**: 3–6× adicional | **Risco**: médio-alto

**Objetivo**: processar N imagens em um único forward pass GPU, multiplicando throughput por fator próximo a N (com retornos decrescentes por VRAM).

### 6.1 Problema: imagens têm tamanhos variados

O `torch.nn.functional.pad` resolve: paddar todas as imagens de um batch para o mesmo tamanho, processar, e cortar os resultados de volta ao tamanho original.

### 6.2 Estratégia de agrupamento

Imagens muito diferentes em tamanho desperdiçam VRAM com padding. Agrupar por tamanho próximo:

```python
def group_tasks_by_size(
    tasks: list[ImageTask],
    batch_size: int,
    size_tolerance: float = 0.2,  # 20% de variação aceitável
) -> list[list[ImageTask]]:
    """Agrupa tasks em lotes de tamanho similar para minimizar padding."""
    import cv2

    # Ler apenas os cabeçalhos para obter dimensões (sem carregar o pixel data)
    sized = []
    for task in tasks:
        img = cv2.imread(task.input_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            sized.append((task, 0, 0))
        else:
            h, w = img.shape[:2]
            sized.append((task, h, w))

    # Ordenar por área (maior → menor)
    sized.sort(key=lambda x: x[1] * x[2], reverse=True)

    groups = []
    current_group = []
    reference_area = None

    for task, h, w in sized:
        area = h * w
        if reference_area is None:
            reference_area = area

        if len(current_group) < batch_size and area >= reference_area * (1 - size_tolerance):
            current_group.append(task)
        else:
            if current_group:
                groups.append(current_group)
            current_group = [task]
            reference_area = area

    if current_group:
        groups.append(current_group)

    return groups
```

### 6.3 Novo método no runner: `upscale_batch`

**Arquivo**: `src/upscale_image/models/base.py`

Adicionar ao contrato `SuperResolutionModel`:

```python
def upscale_batch(
    self,
    images: list[np.ndarray],
    config: AppConfig,
) -> list[np.ndarray]:
    """Upscale um lote de imagens em um único forward pass.

    Implementação default: chama upscale() em loop (compatibilidade total).
    Runners que suportam batch real devem sobrescrever este método.
    """
    return [self.upscale(img, config) for img in images]
```

**Arquivo**: `src/upscale_image/models/realesrgan.py`

Sobrescrever `upscale_batch()` com implementação real:

```python
def upscale_batch(
    self,
    images: list[np.ndarray],
    config: AppConfig,
) -> list[np.ndarray]:
    """Batch inference: N imagens em um único forward pass.

    Todas as imagens são paddadas para o tamanho máximo do batch.
    Os outputs são cortados de volta ao tamanho correto de cada imagem.
    """
    if not self._loaded or self._net is None:
        raise RuntimeError("Model not loaded.")

    device = self._resolve_device(config.runtime.device)
    net    = self._net.to(device).float()

    use_amp = (config.runtime.precision == "fp16") and (device.type == "cuda")

    # Converter todas as imagens para tensor
    tensors = []
    original_sizes = []
    for img in images:
        h, w = img.shape[:2]
        original_sizes.append((h, w))
        rgb = img[:, :, ::-1].copy()
        t = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0)
        tensors.append(t)

    # Paddar para o maior tamanho do batch (padding com zeros = borda preta)
    max_h = max(t.shape[1] for t in tensors)
    max_w = max(t.shape[2] for t in tensors)

    padded = []
    for t in tensors:
        pad_h = max_h - t.shape[1]
        pad_w = max_w - t.shape[2]
        # F.pad: (left, right, top, bottom)
        import torch.nn.functional as F
        p = F.pad(t, (0, pad_w, 0, pad_h), mode="reflect")
        padded.append(p)

    batch_tensor = torch.stack(padded, dim=0).to(device)  # (N, C, H, W)

    with torch.inference_mode():
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            batch_out = net(batch_tensor)  # (N, C, scale*H, scale*W)

    # Cortar outputs de volta ao tamanho original × scale
    results = []
    for i, (orig_h, orig_w) in enumerate(original_sizes):
        out_h = orig_h * self._scale
        out_w = orig_w * self._scale
        out = batch_out[i, :, :out_h, :out_w]
        out_np = (
            out.float().clamp(0.0, 1.0).mul(255.0).round()
               .byte().permute(1, 2, 0).cpu().numpy()
        )
        results.append(out_np[:, :, ::-1].copy())  # RGB → BGR

    return results
```

### 6.4 Integração no pipeline assíncrono

No `async_worker.py`, a thread de inferência processa em grupos ao invés de imagens individuais:

```python
# Na thread infer_loop do async_worker.py:
# Ao invés de processar um _ReadResult por vez, acumular um batch e processar junto.

def infer_loop_batched(batch_size: int):
    pending: list[_ReadResult] = []

    def flush():
        nonlocal pending
        if not pending:
            return
        valid   = [p for p in pending if p.error is None]
        invalid = [p for p in pending if p.error is not None]

        # Erros de leitura → colocar direto na write_queue
        for p in invalid:
            write_queue.put(_InferResult(task=p.task, output=None, ...))

        if valid:
            images = [p.image for p in valid]
            try:
                outputs = model.upscale_batch(images, config)
                for p, out in zip(valid, outputs):
                    h, w = p.image.shape[:2]
                    infer_ms = ...  # tempo total / N como aproximação
                    write_queue.put(_InferResult(task=p.task, output=out, ...))
            except Exception as exc:
                for p in valid:
                    write_queue.put(_InferResult(task=p.task, output=None,
                                                  error=str(exc), ...))
        pending = []

    while True:
        item = read_queue.get()
        if item is _STOP:
            flush()
            write_queue.put(_STOP)
            break
        pending.append(item)
        if len(pending) >= batch_size:
            flush()
```

### 6.5 Gerenciamento dinâmico de VRAM

`batch_size` ideal depende do hardware e tamanho das imagens. Implementar auto-detect opcional:

```python
def estimate_safe_batch_size(
    sample_image: np.ndarray,
    model: SuperResolutionModel,
    config: AppConfig,
    safety_factor: float = 0.7,
) -> int:
    """Estima o batch_size máximo seguro baseado na VRAM disponível."""
    if not torch.cuda.is_available():
        return 1

    total_vram = torch.cuda.get_device_properties(0).total_memory
    free_vram  = total_vram - torch.cuda.memory_allocated(0)
    usable     = int(free_vram * safety_factor)

    h, w, c = sample_image.shape
    # Estimativa conservadora: tensor de input + output + buffers intermediários
    # RRDBNet x4 usa ~8× o tamanho do tensor de input em VRAM durante forward
    bytes_per_image = h * w * c * 4 * 8 * (config.model.scale ** 2)

    return max(1, usable // bytes_per_image)
```

### 6.6 Config e CLI

**`config/schema.py`**: `batch_size: int = 1` (já adicionado na Fase 2)

**`cli/main.py`**:
```python
batch_size: int = typer.Option(1, "--batch-size", "-b",
    help="Images per GPU forward pass. Auto-detect with 0.")
```

---

## 7. Fase 4 — TensorRT via torch-tensorrt

**Esforço**: 2–4 semanas | **Ganho esperado**: 2–4× adicional sobre Python compilado | **Risco**: alto

**Pré-requisito**: GPU NVIDIA com suporte a TensorRT (Turing+, CUDA 12+), `torch-tensorrt==2.11.0` instalado.

### 7.1 Estratégia: novo runner TensorRT, sem alterar o runner existente

O ADR 0004 estabelece que novos modelos/backends são adicionados por registro, sem alterar o pipeline. Segue-se a mesma regra aqui: criar `TensorRTRunner` que carrega um engine compilado.

### 7.2 Script de exportação (offline, uma vez por hardware)

**Novo arquivo**: `scripts/export_tensorrt.py`

```python
"""Exporta um modelo Real-ESRGAN treinado para TensorRT engine.

Uso:
    python scripts/export_tensorrt.py \
        --weights weights/realesrgan-x4.pth \
        --output  weights/realesrgan-x4-trt-fp16.ep \
        --precision fp16 \
        --min-size 64 --opt-size 512 --max-size 2048

O engine resultante é específico para a GPU e versão de TensorRT instalada.
Deve ser regenerado ao trocar de hardware.
"""
import argparse
import torch
import torch_tensorrt
from upscale_image.models.realesrgan import RealESRGANRunner

def export(weights: str, output: str, precision: str,
           min_size: int, opt_size: int, max_size: int):
    runner = RealESRGANRunner(scale=4, weights_path=weights)
    runner.load()
    net = runner._net.cuda().eval()

    dtype = torch.float16 if precision == "fp16" else torch.float32

    # Dynamic shapes: aceita qualquer tamanho entre min e max
    inputs = [
        torch_tensorrt.Input(
            min_shape=[1, 3, min_size, min_size],
            opt_shape=[1, 3, opt_size, opt_size],
            max_shape=[1, 3, max_size, max_size],
            dtype=dtype,
        )
    ]

    trt_model = torch_tensorrt.compile(
        net,
        inputs=inputs,
        enabled_precisions={dtype},
        workspace_size=4 * (1024 ** 3),  # 4 GB
    )

    torch_tensorrt.save(trt_model, output)
    print(f"Engine salvo em: {output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights",   required=True)
    parser.add_argument("--output",    required=True)
    parser.add_argument("--precision", default="fp16", choices=["fp32", "fp16"])
    parser.add_argument("--min-size",  type=int, default=64)
    parser.add_argument("--opt-size",  type=int, default=512)
    parser.add_argument("--max-size",  type=int, default=2048)
    args = parser.parse_args()
    export(args.weights, args.output, args.precision,
           args.min_size, args.opt_size, args.max_size)
```

### 7.3 Novo runner: `src/upscale_image/models/tensorrt_runner.py`

```python
"""Runner que carrega um TensorRT engine compilado (.ep) via torch-tensorrt.

O engine é produzido offline por scripts/export_tensorrt.py.
Implementa o mesmo contrato SuperResolutionModel do runner PyTorch.
"""
from __future__ import annotations

import numpy as np
import torch

from upscale_image.config import AppConfig
from upscale_image.models.base import SuperResolutionModel


class TensorRTRunner(SuperResolutionModel):
    """Runner TensorRT: máxima performance em GPU NVIDIA.

    Args:
        scale:       Fator de upscale (deve coincidir com o engine exportado).
        engine_path: Caminho para o arquivo .ep gerado por export_tensorrt.py.
        precision:   Precisão do engine ("fp16" ou "fp32").
    """

    def __init__(self, scale: int, engine_path: str, precision: str = "fp16") -> None:
        self._scale      = scale
        self._engine_path = engine_path
        self._precision  = precision
        self._model      = None
        self._loaded     = False

    @property
    def name(self) -> str:
        return f"realesrgan-x4-trt-{self._precision}"

    @property
    def scale(self) -> int:
        return self._scale

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        import torch_tensorrt
        from pathlib import Path

        if not Path(self._engine_path).exists():
            raise FileNotFoundError(
                f"TensorRT engine não encontrado: {self._engine_path}. "
                "Execute scripts/export_tensorrt.py primeiro."
            )

        self._model = torch_tensorrt.load(self._engine_path).cuda()
        self._model.eval()

        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True

        self._loaded = True

    def upscale(self, image: np.ndarray, config: AppConfig) -> np.ndarray:
        if not self._loaded:
            raise RuntimeError("Model not loaded.")

        dtype = torch.float16 if self._precision == "fp16" else torch.float32

        img_rgb = image[:, :, ::-1].copy()
        tensor = (
            torch.from_numpy(img_rgb)
            .permute(2, 0, 1)
            .to(dtype)
            .div(255.0)
            .unsqueeze(0)
            .cuda()
        )

        with torch.inference_mode():
            output = self._model(tensor)

        out_np = (
            output.squeeze(0)
            .float().clamp(0.0, 1.0).mul(255.0).round()
            .byte().permute(1, 2, 0).cpu().numpy()
        )
        return out_np[:, :, ::-1].copy()

    def unload(self) -> None:
        self._model = None
        self._loaded = False
        torch.cuda.empty_cache()
```

### 7.4 Registro no registry

**Arquivo**: `src/upscale_image/models/registry.py`

```python
# Adicionar registro condicional: só registra se torch-tensorrt estiver instalado
try:
    from upscale_image.models.tensorrt_runner import TensorRTRunner
    _registry.register(
        "realesrgan-x4-trt-fp16",
        lambda: TensorRTRunner(
            scale=4,
            engine_path="weights/realesrgan-x4-trt-fp16.ep",
            precision="fp16",
        ),
    )
    _registry.register(
        "realesrgan-x4-trt-fp32",
        lambda: TensorRTRunner(
            scale=4,
            engine_path="weights/realesrgan-x4-trt-fp32.ep",
            precision="fp32",
        ),
    )
except ImportError:
    pass  # torch-tensorrt não instalado — runners TRT não disponíveis
```

### 7.5 Suporte a ONNX Runtime (alternativa cross-vendor)

Para ambientes sem GPU NVIDIA (AMD, CPU), criar `OnnxRunner`:

**Novo arquivo**: `src/upscale_image/models/onnx_runner.py`

```python
"""Runner ONNX Runtime: suporte a CUDA EP, TensorRT EP e CPU EP.

Permite executar o modelo em AMD GPU (ROCm), Intel (OpenVINO EP),
ou CPU (OpenMP) sem dependência de torch-tensorrt.

Exportar o modelo:
    torch.onnx.export(model, dummy_input, "weights/realesrgan-x4.onnx",
                      dynamic_axes={"input": {0: "N", 2: "H", 3: "W"},
                                    "output": {0: "N", 2: "H", 3: "W"}})
"""
from __future__ import annotations

import numpy as np

from upscale_image.config import AppConfig
from upscale_image.models.base import SuperResolutionModel


class OnnxRunner(SuperResolutionModel):
    def __init__(self, scale: int, onnx_path: str, provider: str = "CUDAExecutionProvider"):
        self._scale    = scale
        self._onnx     = onnx_path
        self._provider = provider
        self._session  = None
        self._loaded   = False

    @property
    def name(self) -> str:
        return f"realesrgan-x4-onnx-{self._provider.replace('ExecutionProvider', '').lower()}"

    @property
    def scale(self) -> int:
        return self._scale

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        import onnxruntime as ort
        from pathlib import Path

        if not Path(self._onnx).exists():
            raise FileNotFoundError(f"ONNX model não encontrado: {self._onnx}")

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 4

        self._session = ort.InferenceSession(
            self._onnx,
            sess_options=opts,
            providers=[self._provider, "CPUExecutionProvider"],
        )
        self._loaded = True

    def upscale(self, image: np.ndarray, config: AppConfig) -> np.ndarray:
        if not self._loaded:
            raise RuntimeError("Model not loaded.")

        img_rgb = image[:, :, ::-1].copy()
        tensor  = img_rgb.transpose(2, 0, 1).astype(np.float32)[np.newaxis] / 255.0

        input_name  = self._session.get_inputs()[0].name
        output_name = self._session.get_outputs()[0].name
        [output]    = self._session.run([output_name], {input_name: tensor})

        out = (output[0].transpose(1, 2, 0).clip(0, 1) * 255).round().astype(np.uint8)
        return out[:, :, ::-1].copy()

    def unload(self) -> None:
        self._session = None
        self._loaded  = False
```

**Script de exportação ONNX**: `scripts/export_onnx.py`

```python
"""Exporta Real-ESRGAN para formato ONNX com dynamic shapes.

Uso:
    python scripts/export_onnx.py \
        --weights weights/realesrgan-x4.pth \
        --output  weights/realesrgan-x4.onnx
"""
import argparse
import torch
from upscale_image.models.realesrgan import RealESRGANRunner

def export(weights: str, output: str):
    runner = RealESRGANRunner(scale=4, weights_path=weights)
    runner.load()
    net = runner._net.eval()

    dummy = torch.randn(1, 3, 256, 256)
    torch.onnx.export(
        net, dummy, output,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input":  {0: "batch", 2: "height", 3: "width"},
            "output": {0: "batch", 2: "height", 3: "width"},
        },
        opset_version=17,
    )
    print(f"ONNX exportado: {output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output",  required=True)
    args = parser.parse_args()
    export(args.weights, args.output)
```

---

## 8. Fase 5 — Multi-GPU com Worker Pool

**Esforço**: 3–4 semanas | **Ganho esperado**: N× (linear por GPU) | **Risco**: alto

**Premissa**: máquina com ≥2 GPUs NVIDIA. Cada GPU recebe um processo dedicado com sua própria cópia do modelo.

### 8.1 Arquitetura

```
Processo Principal
├── Descoberta de imagens → lista de tasks
├── Distribui tasks em multiprocessing.Queue
└── Aguarda resultados em results_queue

Processo GPU:0           Processo GPU:1           Processo GPU:N
├── Carrega modelo       ├── Carrega modelo       ├── Carrega modelo
├── Consome da queue     ├── Consome da queue     ├── Consome da queue
└── Produz resultados    └── Produz resultados    └── Produz resultados
```

### 8.2 Novo módulo: `src/upscale_image/pipeline/multi_gpu.py`

```python
"""Multi-GPU worker pool: distribui tasks entre múltiplas GPUs.

Usa multiprocessing para isolamento real de memória e GIL por GPU.
Cada worker carrega sua própria cópia do modelo na GPU atribuída.

Compatível com todos os runners que implementam SuperResolutionModel.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import time
from typing import Callable

import torch

from upscale_image.config import AppConfig
from upscale_image.io.task import ImageTask
from upscale_image.models.base import SuperResolutionModel
from upscale_image.pipeline.batch import ItemResult


def _gpu_worker(
    gpu_id: int,
    model_factory: Callable[[], SuperResolutionModel],
    task_queue: "mp.Queue[ImageTask | None]",
    result_queue: "mp.Queue[ItemResult]",
    config: AppConfig,
) -> None:
    """Worker process: carrega modelo na GPU gpu_id e processa tasks da fila."""
    import cv2
    from pathlib import Path

    # Isolar este processo para a GPU designada
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Recriar config com device correto
    config.runtime.device = "cuda"

    model = model_factory()
    model.load()

    while True:
        task = task_queue.get()
        if task is None:  # poison pill — encerrar
            break

        t0 = time.monotonic()
        try:
            image = cv2.imread(task.input_path)
            if image is None:
                raise RuntimeError(f"cv2.imread returned None for {task.input_path!r}")

            h, w = image.shape[:2]
            t_infer = time.monotonic()
            output = model.upscale(image, config)
            infer_ms = (time.monotonic() - t_infer) * 1000.0

            out_h, out_w = output.shape[:2]
            Path(task.output_path).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(task.output_path, output)

            task.status = "done"
            result_queue.put(ItemResult(
                task=task, status="done",
                elapsed=time.monotonic() - t0,
                inference_time_ms=infer_ms,
                input_width=w, input_height=h,
                output_width=out_w, output_height=out_h,
            ))

        except Exception as exc:  # noqa: BLE001
            task.status = "failed"
            result_queue.put(ItemResult(
                task=task, status="failed",
                elapsed=time.monotonic() - t0,
                error=str(exc),
            ))

    model.unload()


def run_batch_multi_gpu(
    config: AppConfig,
    tasks: list[ImageTask],
    model_factory: Callable[[], SuperResolutionModel],
    gpu_ids: list[int] | None = None,
) -> list[ItemResult]:
    """Distribui tasks entre múltiplas GPUs em processos paralelos.

    Args:
        config:        AppConfig base (device será sobrescrito por GPU id).
        tasks:         Lista ordenada de ImageTask.
        model_factory: Callable sem args que retorna um runner não-carregado.
        gpu_ids:       IDs das GPUs a usar. None = detectar automaticamente.

    Returns:
        Lista de ItemResult na mesma ordem de tasks.
    """
    if gpu_ids is None:
        gpu_ids = list(range(torch.cuda.device_count()))

    if not gpu_ids:
        raise RuntimeError("Nenhuma GPU disponível para multi-GPU.")

    ctx = mp.get_context("spawn")  # spawn para isolamento CUDA
    task_queue:   "mp.Queue" = ctx.Queue()
    result_queue: "mp.Queue" = ctx.Queue()

    # Enfileirar todas as tasks
    for task in tasks:
        task_queue.put(task)

    # Poison pills para encerrar workers
    for _ in gpu_ids:
        task_queue.put(None)

    # Iniciar workers
    workers = []
    for gpu_id in gpu_ids:
        p = ctx.Process(
            target=_gpu_worker,
            args=(gpu_id, model_factory, task_queue, result_queue, config),
            daemon=True,
        )
        p.start()
        workers.append(p)

    # Coletar resultados
    results_map: dict[str, ItemResult] = {}
    for _ in tasks:
        result = result_queue.get()
        results_map[result.task.input_path] = result

    for p in workers:
        p.join()

    # Retornar na ordem original
    return [results_map[t.input_path] for t in tasks]
```

### 8.3 Integração no CLI

**`cli/main.py`**:
```python
multi_gpu: bool = typer.Option(False, "--multi-gpu",
    help="Distribute across all available GPUs (requires CUDA)")
```

**`batch.py`** — adicionar branch para multi-GPU:
```python
if config.runtime.multi_gpu and torch.cuda.device_count() > 1:
    from upscale_image.pipeline.multi_gpu import run_batch_multi_gpu
    results = run_batch_multi_gpu(config, discovery.tasks, model_factory)
```

---

## 9. Adição do Modelo SwinIR

**Esforço**: 1 semana | **Impacto**: qualidade superior, menos artefatos em texto/PDF

### 9.1 Dependência

```
# requirements/swinir.txt (novo)
-r base.txt
timm>=1.0.0    # SwinIR usa Vision Transformer components via timm
```

### 9.2 Novo runner: `src/upscale_image/models/swinir_runner.py`

SwinIR pode ser carregado a partir dos pesos oficiais `003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth` disponíveis no repositório oficial.

```python
"""Runner SwinIR: Swin Transformer para super-resolução.

Vantagens sobre Real-ESRGAN:
- Menos alucinações em imagens com texto (ideal para PDFs)
- Bordas mais nítidas e geometricamente precisas
- Menos artefatos em regiões de baixa textura

Pesos: baixar de https://github.com/JingyunLiang/SwinIR/releases
Colocar em: weights/swinir-x4.pth
"""
from __future__ import annotations

import numpy as np
import torch

from upscale_image.config import AppConfig
from upscale_image.models.base import SuperResolutionModel


class SwinIRRunner(SuperResolutionModel):
    """Runner para SwinIR Real-SR (Large, x4)."""

    def __init__(self, scale: int = 4, weights_path: str = "weights/swinir-x4.pth"):
        self._scale        = scale
        self._weights_path = weights_path
        self._net          = None
        self._loaded       = False

    @property
    def name(self) -> str:
        return "swinir-x4"

    @property
    def scale(self) -> int:
        return self._scale

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        from pathlib import Path

        # SwinIR architecture — importar do repositório clonado localmente
        # ou instalar via: pip install git+https://github.com/JingyunLiang/SwinIR
        try:
            from swinir import SwinIR as SwinIRNet
        except ImportError:
            raise ImportError(
                "SwinIR não instalado. Instale com: "
                "pip install git+https://github.com/JingyunLiang/SwinIR.git"
            )

        if not Path(self._weights_path).exists():
            raise FileNotFoundError(f"Pesos SwinIR não encontrados: {self._weights_path}")

        # Configuração para Real-SR Large x4
        net = SwinIRNet(
            upscale=self._scale,
            in_chans=3,
            img_size=64,
            window_size=8,
            img_range=1.0,
            depths=[6, 6, 6, 6, 6, 6, 6, 6, 6],
            embed_dim=240,
            num_heads=[8, 8, 8, 8, 8, 8, 8, 8, 8],
            mlp_ratio=2,
            upsampler="nearest+conv",
            resi_connection="3conv",
        )

        state_dict = torch.load(self._weights_path, map_location="cpu", weights_only=True)
        if "params_ema" in state_dict:
            state_dict = state_dict["params_ema"]
        elif "params" in state_dict:
            state_dict = state_dict["params"]

        net.load_state_dict(state_dict, strict=True)
        net.eval()

        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            net = torch.compile(net, mode="reduce-overhead")

        self._net    = net
        self._loaded = True

    def upscale(self, image: np.ndarray, config: AppConfig) -> np.ndarray:
        if not self._loaded or self._net is None:
            raise RuntimeError("Model not loaded.")

        from upscale_image.models.realesrgan import RealESRGANRunner
        device  = RealESRGANRunner._resolve_device(config.runtime.device)
        net     = self._net.to(device)
        use_amp = (config.runtime.precision == "fp16") and (device.type == "cuda")

        img_rgb = image[:, :, ::-1].copy()
        tensor  = (
            torch.from_numpy(img_rgb)
            .permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)
        )

        # SwinIR requer que H e W sejam múltiplos de window_size (8)
        window_size = 8
        _, _, h, w = tensor.shape
        pad_h = (window_size - h % window_size) % window_size
        pad_w = (window_size - w % window_size) % window_size
        import torch.nn.functional as F
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")

        with torch.inference_mode():
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                output = net(tensor)

        # Remover padding e converter
        output = output[:, :, :h * self._scale, :w * self._scale]
        out_np = (
            output.squeeze(0).float().clamp(0, 1).mul(255).round()
                  .byte().permute(1, 2, 0).cpu().numpy()
        )
        return out_np[:, :, ::-1].copy()

    def unload(self) -> None:
        self._net    = None
        self._loaded = False
        torch.cuda.empty_cache()
```

### 9.3 Registro

**`src/upscale_image/models/registry.py`**:
```python
try:
    from upscale_image.models.swinir_runner import SwinIRRunner
    _registry.register("swinir-x4", lambda: SwinIRRunner(scale=4))
except ImportError:
    pass  # SwinIR não instalado
```

---

## 10. Alterações em Schema e Config

Resumo de todas as adições ao `RuntimeConfig` ao longo das 4 fases:

```python
@dataclass
class RuntimeConfig:
    # ── Existente ─────────────────────────────────────────
    device:     str                        = "cpu"
    precision:  Literal["fp32", "fp16"]   = "fp32"
    tile_size:  int                        = 0
    tile_pad:   int                        = 32      # Fase 1: 10 → 32

    # ── Fase 2: Pipeline Assíncrono ───────────────────────
    async_io:       bool = False   # habilita I/O assíncrono
    prefetch_size:  int  = 4       # buffer de leitura antecipada
    write_workers:  int  = 2       # threads de escrita paralela

    # ── Fase 3: Batch Inference ───────────────────────────
    batch_size:     int  = 1       # 0 = auto-detect por VRAM

    # ── Fase 5: Multi-GPU ─────────────────────────────────
    multi_gpu:      bool      = False
    gpu_ids:        list[int] = field(default_factory=list)  # [] = usar todas
```

---

## 11. Impacto nos Testes Existentes

### Testes que precisam ser atualizados

| Arquivo de teste | Motivo |
|---|---|
| `tests/test_config.py` | `tile_pad` default mudou de 10 → 32 |
| `tests/test_realesrgan_runner.py` | `isinstance(._net, RRDBNet)` quebra após `torch.compile`; remover verificação de tipo interno |
| `tests/test_batch.py` | Novo parâmetro `async_io` em `run_batch()`; testar que comportamento serial não mudou |

### Novos testes a criar

| Arquivo | O que testar |
|---|---|
| `tests/test_async_worker.py` | Pipeline assíncrono: ordem preservada, erros isolados, shutdown limpo |
| `tests/test_batch_inference.py` | `upscale_batch()`: dimensões corretas, erros por item |
| `tests/test_tensorrt_runner.py` | Carregamento de engine, inferência, `FileNotFoundError` correto (usar mock) |
| `tests/test_swinir_runner.py` | Padding para window_size, output shape correto |
| `tests/test_multi_gpu.py` | Distribuição de tasks, ordem do resultado (usar mock de GPU) |

### Estratégia para testes de GPU sem GPU disponível

Usar `unittest.mock.patch` para simular `torch.cuda.is_available()` e substituir o modelo por `MockRunner`. Os testes de performance e compilação de modelo devem ser marcados com `@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")`.

---

## 12. Novos Arquivos a Criar

```
src/upscale_image/
├── models/
│   ├── swinir_runner.py          # Fase SwinIR
│   ├── tensorrt_runner.py        # Fase 4
│   └── onnx_runner.py            # Fase 4 (alternativa cross-vendor)
├── pipeline/
│   └── async_worker.py           # Fase 2
│   └── multi_gpu.py              # Fase 5

scripts/
├── export_onnx.py                # Fase 4: exportar para ONNX
└── export_tensorrt.py            # Fase 4: compilar engine TensorRT

requirements/
├── performance.txt               # torch-tensorrt
└── swinir.txt                    # timm para SwinIR

docs/adr/
├── 0011-compilacao-de-modelo-e-amp-automatico.md
├── 0012-batch-inference.md
├── 0013-pipeline-assincrono-prefetch.md
└── 0014-backend-de-inferencia-plugavel.md

tests/
├── test_async_worker.py
├── test_batch_inference.py
├── test_tensorrt_runner.py
├── test_swinir_runner.py
└── test_multi_gpu.py
```

---

## 13. Resumo de Ganhos por Fase

| Fase | Técnica | Ganho Isolado | Ganho Acumulado | Esforço |
|---|---|---|---|---|
| Baseline | Código atual | 1× | 1× | — |
| 1 | cudnn.benchmark + torch.compile + AMP + tile_pad | 2–3× | 2–3× | 1–2 dias |
| 2 | Pipeline assíncrono + prefetch | 1.3–1.8× | 3–5× | 1 semana |
| 3 | Batch inference (batch=4) | 2–4× | 7–15× | 1–2 semanas |
| 4 | TensorRT FP16 | 2–4× | 15–50× | 2–4 semanas |
| 5 | Multi-GPU ×4 | ~4× | 60–200× | 3–4 semanas |

> Ganhos são multiplicativos e estimados para RTX 3080, imagens 1080p, Real-ESRGAN x4. Hardware diferente produzirá resultados diferentes. A Fase 1 é a de maior ROI e deve ser implementada antes de qualquer outra.

---

## 14. Ordem de Implementação Recomendada

```
Semana 1:
  ├── Fase 1 completa (2–3 dias)
  │   ├── tile_pad: 10 → 32
  │   ├── Gaussian feathering no tiling
  │   ├── cudnn.benchmark na inicialização
  │   ├── torch.compile no load()
  │   └── AMP autocast substituindo half() manual
  └── Atualizar ADR 0007 e criar ADR 0011

Semana 2–3:
  ├── Fase 2: async_worker.py + integração em batch.py
  ├── Testes test_async_worker.py
  └── Criar ADR 0013

Semana 4–5:
  ├── Fase 3: upscale_batch() + agrupamento por tamanho
  ├── Testes test_batch_inference.py
  ├── Atualizar ADR 0005 e criar ADR 0012
  └── Auto-detect batch_size por VRAM

Semana 6–7:
  ├── SwinIR runner + registro condicional
  └── Testes test_swinir_runner.py

Semana 8–11:
  ├── Fase 4: export_onnx.py + onnx_runner.py
  ├── Fase 4: export_tensorrt.py + tensorrt_runner.py
  ├── Testes test_tensorrt_runner.py
  └── Criar ADR 0014

Semana 12–15:
  ├── Fase 5: multi_gpu.py
  ├── Testes test_multi_gpu.py
  └── Integração end-to-end multi-GPU
```
