# Prompt do Passo 28

Execute o passo 28 do projeto: implementar backend de inferência plugável com TensorRT e ONNX Runtime.

## Leia antes de implementar

- `/home/edgar/dev/upscale_image/docs/parts/28-fase4-backend-inferencia-plugavel.md`
- `/home/edgar/dev/upscale_image/docs/adr/0014-backend-de-inferencia-plugavel.md`
- `/home/edgar/dev/upscale_image/docs/adr/0004-contrato-de-modelo-e-registry.md`
- `/home/edgar/dev/upscale_image/src/upscale_image/models/base.py`
- `/home/edgar/dev/upscale_image/src/upscale_image/models/registry.py`

## Regras fixas

- `torch-tensorrt` e `onnxruntime-gpu` são **mutuamente exclusivos** — não instalar juntos;
- registros de `TensorRTRunner` e `OnnxRunner` em `registry.py` devem ser **condicionais** (`try/except ImportError`);
- não importar `torch_tensorrt` ou `onnxruntime` no top-level dos módulos — apenas dentro de `load()`;
- o engine TensorRT (`.ep`) é hardware-specific — não distribuir como artefato portável;
- scripts de exportação em `scripts/` devem ter interface de linha de comando clara.

## Sequência mínima esperada

1. Criar `requirements/performance.txt` e atualizar `pyproject.toml` com extras `performance` e `onnx`.
2. Criar `src/upscale_image/models/tensorrt_runner.py`.
3. Criar `src/upscale_image/models/onnx_runner.py`.
4. Criar `scripts/export_tensorrt.py` e `scripts/export_onnx.py`.
5. Atualizar `registry.py` com registros condicionais para TRT (fp16, fp32) e ONNX (cuda, cpu).
6. Criar `tests/test_tensorrt_runner.py` e `tests/test_onnx_runner.py` usando `unittest.mock`.
7. Executar `pytest tests/test_tensorrt_runner.py tests/test_onnx_runner.py -v`.
8. Executar `pytest tests/` e garantir regressão completa.

## Antes de finalizar

- executar `pytest tests/test_tensorrt_runner.py tests/test_onnx_runner.py tests/test_registry.py -v`;
- confirmar que `pytest tests/` passa completamente;
- atualizar `prompt/control.yaml`.
