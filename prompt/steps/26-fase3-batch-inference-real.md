# Prompt do Passo 26

Execute o passo 26 do projeto: implementar batch inference real (batch_size > 1).

## Leia antes de implementar

- `/home/edgar/dev/upscale_image/docs/parts/26-fase3-batch-inference-real.md`
- `/home/edgar/dev/upscale_image/docs/adr/0012-batch-inference.md`
- `/home/edgar/dev/upscale_image/docs/adr/0004-contrato-de-modelo-e-registry.md`
- `/home/edgar/dev/upscale_image/docs/adr/0005-pipeline-deterministico-e-estrategia-de-erros.md`
- `/home/edgar/dev/upscale_image/src/upscale_image/models/base.py`
- `/home/edgar/dev/upscale_image/src/upscale_image/models/realesrgan.py`
- `/home/edgar/dev/upscale_image/src/upscale_image/pipeline/async_worker.py`
- `/home/edgar/dev/upscale_image/src/upscale_image/pipeline/batch.py`

## Regras fixas

- `upscale_batch()` tem implementação default em `base.py` (loop sobre `upscale()`) — **todos os runners existentes continuam funcionando sem alteração**;
- outputs de `upscale_batch()` devem ter shape `(orig_h * scale, orig_w * scale, 3)` — nunca retornar tensors padded;
- `batch_size=0` significa auto-detect por VRAM disponível;
- falhas em `upscale_batch()` seguem ADR 0005: cada item do batch com erro gera `ItemResult(status="failed")`.

## Sequência mínima esperada

1. Adicionar `upscale_batch()` com implementação default em `models/base.py`.
2. Sobrescrever `upscale_batch()` em `models/realesrgan.py` com padding, forward único, corte.
3. Implementar `group_tasks_by_size()` e `estimate_safe_batch_size()` em `pipeline/batch.py`.
4. Atualizar `pipeline/async_worker.py` para processar em batch quando `batch_size > 1`.
5. Adicionar `batch_size: int = 1` em `config/schema.py`.
6. Adicionar `--batch-size` em `cli/main.py`.
7. Criar `tests/test_batch_inference.py`.
8. Executar `pytest tests/` e garantir regressão completa.

## Antes de finalizar

- executar `pytest tests/test_batch_inference.py tests/test_async_worker.py -v`;
- confirmar que `pytest tests/` passa completamente;
- atualizar `prompt/control.yaml`.
