# Passo 28: Fase 4 — Backend de Inferência Plugável (TensorRT / ONNX Runtime)

## Objetivo

Abstrair o backend de execução do modelo atrás da interface `SuperResolutionModel`
existente (ADR 0004), adicionando dois novos runners opcionais:

- **`TensorRTRunner`**: máxima performance em GPU NVIDIA via `torch-tensorrt`. Ganho
  esperado: **2–4× adicional** sobre PyTorch compilado (Fase 1).
- **`OnnxRunner`**: suporte cross-vendor (AMD, Intel, CPU) via `onnxruntime`. Sem
  dependência de `torch-tensorrt`.

O pipeline não muda — apenas novos runners são registrados condicionalmente.

## ADRs relacionadas

- [0004-contrato-de-modelo-e-registry.md](../adr/0004-contrato-de-modelo-e-registry.md) — novos backends são adicionados por registro, sem alterar o pipeline
- [0014-backend-de-inferencia-plugavel.md](../adr/0014-backend-de-inferencia-plugavel.md) — decisão de abstração do backend e exclusão mútua `torch-tensorrt` vs `onnxruntime-gpu`
- [0007-stack-tecnologica-principal.md](../adr/0007-stack-tecnologica-principal.md) — `torch-tensorrt` e `onnxruntime-gpu` são dependências opcionais

## Dependências

- Passo 27 concluído (padrão de registro condicional estabelecido com SwinIR)
- GPU NVIDIA com TensorRT support (Turing+, CUDA 12+) para testar `TensorRTRunner` em hardware real
- Pré-requisito operacional: `torch-tensorrt==2.11.0` alinhado com `torch==2.11.0` (mesma versão major)

## Entregáveis

1. `src/upscale_image/models/tensorrt_runner.py` — runner TensorRT carregando engine `.ep`
2. `src/upscale_image/models/onnx_runner.py` — runner ONNX Runtime com suporte a `CUDAExecutionProvider` e `CPUExecutionProvider`
3. `scripts/export_tensorrt.py` — script offline de exportação do engine TensorRT
4. `scripts/export_onnx.py` — script offline de exportação do modelo ONNX
5. `requirements/performance.txt` — dependências opcionais de performance
6. `pyproject.toml` atualizado com optional extras `performance` e `onnx`
7. `src/upscale_image/models/registry.py` atualizado com registros condicionais de TRT e ONNX
8. `tests/test_tensorrt_runner.py` cobrindo o contrato via mocks
9. `tests/test_onnx_runner.py` cobrindo o contrato via mocks

## Escopo obrigatório

### 7.1 Arquivo de dependências `requirements/performance.txt`

```
-r base.txt
torch-tensorrt==2.11.0
```

> **Atenção**: `torch-tensorrt` e `onnxruntime-gpu` não devem coexistir no mesmo ambiente
> (conflitos de CUDA runtime). Criar o extras ONNX separado em `pyproject.toml`.

**`pyproject.toml`** — adicionar optional extras:

```toml
[project.optional-dependencies]
performance = ["torch-tensorrt==2.11.0"]
onnx        = ["onnxruntime-gpu==1.20.0"]
```

### 7.2 Runner `TensorRTRunner`

**Arquivo a criar**: `src/upscale_image/models/tensorrt_runner.py`

Carrega um engine TensorRT compilado (`.ep`) gerado offline por `scripts/export_tensorrt.py`.

Contrato:
- `name` → `f"realesrgan-x4-trt-{precision}"` (ex: `"realesrgan-x4-trt-fp16"`)
- `scale` → fator de upscale configurado
- `load()` → `torch_tensorrt.load(engine_path).cuda().eval()`; falha com `FileNotFoundError` se `.ep` não existir
- `upscale()` → converte imagem para tensor no dtype do engine, forward pass, converte de volta
- `unload()` → libera modelo, `torch.cuda.empty_cache()`

**Tratamento de importação**:

```python
def load(self) -> None:
    try:
        import torch_tensorrt
    except ImportError:
        raise ImportError(
            "torch-tensorrt não instalado. "
            "Instale com: pip install -r requirements/performance.txt"
        )
```

### 7.3 Script de exportação TensorRT

**Arquivo a criar**: `scripts/export_tensorrt.py`

Script de linha de comando, executado **offline uma vez por hardware**. Gera um engine
`.ep` específico para a GPU e versão de TensorRT instalada — deve ser regenerado ao
trocar de hardware.

Interface esperada:

```
python scripts/export_tensorrt.py \
    --weights weights/realesrgan-x4.pth \
    --output  weights/realesrgan-x4-trt-fp16.ep \
    --precision fp16 \
    --min-size 64 --opt-size 512 --max-size 2048
```

Usa `torch_tensorrt.Input` com `min_shape`, `opt_shape`, `max_shape` para dynamic shapes.

### 7.4 Runner `OnnxRunner`

**Arquivo a criar**: `src/upscale_image/models/onnx_runner.py`

Runner alternativo sem dependência de `torch`. Usa `onnxruntime` com suporte a:
- `CUDAExecutionProvider` — GPU NVIDIA via CUDA
- `CPUExecutionProvider` — fallback automático

Contrato:
- `name` → `f"realesrgan-x4-onnx-{provider_suffix}"` (ex: `"realesrgan-x4-onnx-cuda"`)
- `load()` → `ort.InferenceSession(onnx_path, providers=[provider, "CPUExecutionProvider"])`; falha com `FileNotFoundError` se `.onnx` não existir
- `upscale()` → numpy pipeline: `(H, W, C)` → `(1, C, H, W)` float32 / 255 → run session → `(H*scale, W*scale, C)` uint8
- `unload()` → `self._session = None`

**Tratamento de importação**:

```python
def load(self) -> None:
    try:
        import onnxruntime as ort
    except ImportError:
        raise ImportError(
            "onnxruntime não instalado. "
            "Instale com: pip install onnxruntime-gpu==1.20.0"
        )
```

### 7.5 Script de exportação ONNX

**Arquivo a criar**: `scripts/export_onnx.py`

Exporta o modelo para formato ONNX com dynamic axes para batch, height e width.

Interface esperada:

```
python scripts/export_onnx.py \
    --weights weights/realesrgan-x4.pth \
    --output  weights/realesrgan-x4.onnx
```

Usa `torch.onnx.export()` com `opset_version=17` e `dynamic_axes`.

### 7.6 Registros condicionais no registry

**Arquivo**: `src/upscale_image/models/registry.py`

Adicionar ao final (após SwinIR):

```python
# TensorRT runners
try:
    from upscale_image.models.tensorrt_runner import TensorRTRunner
    _registry.register(
        "realesrgan-x4-trt-fp16",
        lambda: TensorRTRunner(scale=4,
                               engine_path="weights/realesrgan-x4-trt-fp16.ep",
                               precision="fp16"),
    )
    _registry.register(
        "realesrgan-x4-trt-fp32",
        lambda: TensorRTRunner(scale=4,
                               engine_path="weights/realesrgan-x4-trt-fp32.ep",
                               precision="fp32"),
    )
except ImportError:
    pass  # torch-tensorrt não instalado

# ONNX runner
try:
    from upscale_image.models.onnx_runner import OnnxRunner
    _registry.register(
        "realesrgan-x4-onnx-cuda",
        lambda: OnnxRunner(scale=4,
                           onnx_path="weights/realesrgan-x4.onnx",
                           provider="CUDAExecutionProvider"),
    )
    _registry.register(
        "realesrgan-x4-onnx-cpu",
        lambda: OnnxRunner(scale=4,
                           onnx_path="weights/realesrgan-x4.onnx",
                           provider="CPUExecutionProvider"),
    )
except ImportError:
    pass  # onnxruntime não instalado
```

## Fora de escopo

- Integração com `torch-tensorrt` via Ray (é a Fase 5)
- Suporte a AMD ROCm via ONNX EP (pode ser adicionado registrando outro `OnnxRunner`)
- TensorRT para SwinIR (requer engine exportado separado)

## Novos testes a criar

**Arquivo**: `tests/test_tensorrt_runner.py`

Cobrir (todos via mocks — não requer GPU real):
- `TensorRTRunner` implementa `SuperResolutionModel`
- `load()` falha com `FileNotFoundError` quando engine não existe
- `load()` falha com `ImportError` claro quando `torch_tensorrt` não instalado (mock `ImportError`)
- `upscale()` falha com `RuntimeError` quando chamado sem `load()`
- `unload()` limpa o modelo; `is_loaded` retorna `False`
- Registro condicional: `"realesrgan-x4-trt-fp16"` em `registry.available()` quando TRT disponível

**Arquivo**: `tests/test_onnx_runner.py`

Cobrir (todos via mocks):
- `OnnxRunner` implementa `SuperResolutionModel`
- `load()` falha com `FileNotFoundError` quando `.onnx` não existe
- `load()` falha com `ImportError` claro quando `onnxruntime` não instalado
- `upscale()` retorna array com shape `(H * scale, W * scale, 3)` e dtype `uint8` (mock session)
- `unload()` limpa a sessão; `is_loaded` retorna `False`

## Sequência sugerida para implementação

1. Criar `requirements/performance.txt` e atualizar `pyproject.toml`.
2. Criar `src/upscale_image/models/tensorrt_runner.py`.
3. Criar `src/upscale_image/models/onnx_runner.py`.
4. Criar `scripts/export_tensorrt.py` e `scripts/export_onnx.py`.
5. Atualizar `registry.py` com registros condicionais.
6. Criar `tests/test_tensorrt_runner.py` e `tests/test_onnx_runner.py` usando `unittest.mock`.
7. Executar `pytest tests/test_tensorrt_runner.py tests/test_onnx_runner.py -v`.
8. Executar `pytest tests/` para regressão completa.

## Critérios de aceite

- `TensorRTRunner` e `OnnxRunner` implementam o contrato `SuperResolutionModel` completo
- Ambos falham com mensagens claras para arquivo ausente (`FileNotFoundError`) e pacote ausente (`ImportError`)
- Registry registra runners TRT e ONNX condicionalmente (sem quebrar quando não instalados)
- `requirements/performance.txt` existe; `pyproject.toml` tem extras `performance` e `onnx`
- Scripts de exportação em `scripts/` com interface de linha de comando documentada
- `pytest tests/` passa completamente

## Como testar

```bash
pytest tests/test_tensorrt_runner.py tests/test_onnx_runner.py -v
pytest tests/ -q   # regressão completa
```

## Armadilhas a evitar

- Não instalar `torch-tensorrt` e `onnxruntime-gpu` no mesmo ambiente — conflitos de CUDA runtime
- Não registrar runners TRT/ONNX sem `try/except ImportError` — quebra o registry em ambientes sem as libs
- Não tentar importar `torch_tensorrt` no top-level do módulo — só dentro de `load()`
- O engine TensorRT é hardware-specific — não distribuir `.ep` como artefato portável
- O modelo ONNX tem dynamic axes — não fixar batch, height ou width na exportação
