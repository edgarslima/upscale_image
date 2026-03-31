# Prompt do Passo 25

Execute o passo 25 do projeto: implementar pipeline assíncrono com prefetch de I/O.

## Leia antes de implementar

- `/home/edgar/dev/upscale_image/docs/parts/25-fase2-pipeline-assincrono-com-prefetch.md`
- `/home/edgar/dev/upscale_image/docs/adr/0013-pipeline-assincrono-prefetch.md`
- `/home/edgar/dev/upscale_image/docs/adr/0005-pipeline-deterministico-e-estrategia-de-erros.md`
- `/home/edgar/dev/upscale_image/docs/adr/0003-configuracao-com-precedencia-cli-yaml-defaults.md`
- `/home/edgar/dev/upscale_image/src/upscale_image/pipeline/batch.py`
- `/home/edgar/dev/upscale_image/src/upscale_image/config/schema.py`
- `/home/edgar/dev/upscale_image/src/upscale_image/cli/main.py`

## Regras fixas

- o comportamento serial (`async_io=False`) deve permanecer **identicamente igual** ao pré-existente;
- `run_batch_async()` retorna resultados **na mesma ordem** que `tasks`;
- falhas individuais em qualquer estágio (leitura, inferência, escrita) geram `ItemResult(status="failed")` sem derrubar a run;
- usar `threading.Thread` e `queue.Queue` (stdlib) — sem novas dependências;
- cada write worker deve receber exatamente um sentinel `_STOP`.

## Sequência mínima esperada

1. Criar `src/upscale_image/pipeline/async_worker.py` com dataclasses `_ReadResult`, `_InferResult`, sentinel `_STOP` e função `run_batch_async()`.
2. Implementar threads de leitura, inferência e pool de escrita.
3. Adicionar parâmetros `async_io`, `prefetch_size`, `write_workers` em `pipeline/batch.py`.
4. Adicionar campos `async_io`, `prefetch_size`, `write_workers` em `config/schema.py`.
5. Adicionar flags `--async-io` e `--prefetch` em `cli/main.py`.
6. Criar `tests/test_async_worker.py` cobrindo ordem, isolamento de falhas e shutdown limpo.
7. Executar `pytest tests/` e garantir regressão completa.

## Antes de finalizar

- executar `pytest tests/test_async_worker.py tests/test_batch.py -v`;
- confirmar que `pytest tests/` passa completamente;
- atualizar `prompt/control.yaml`.
