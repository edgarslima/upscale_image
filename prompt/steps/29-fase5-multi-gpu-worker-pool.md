# Prompt do Passo 29

Execute o passo 29 do projeto: implementar multi-GPU com worker pool usando multiprocessing.

## Leia antes de implementar

- `/home/edgar/dev/upscale_image/docs/parts/29-fase5-multi-gpu-worker-pool.md`
- `/home/edgar/dev/upscale_image/docs/adr/0005-pipeline-deterministico-e-estrategia-de-erros.md`
- `/home/edgar/dev/upscale_image/docs/adr/0013-pipeline-assincrono-prefetch.md`
- `/home/edgar/dev/upscale_image/docs/adr/0001-arquitetura-cli-local-monolitica.md`
- `/home/edgar/dev/upscale_image/src/upscale_image/pipeline/batch.py`
- `/home/edgar/dev/upscale_image/src/upscale_image/config/schema.py`
- `/home/edgar/dev/upscale_image/src/upscale_image/cli/main.py`

## Regras fixas

- usar `mp.get_context("spawn")` — nunca `"fork"` com CUDA;
- o modelo não deve ser carregado no processo principal e passado para workers — cada worker instancia e carrega o seu próprio modelo;
- `run_batch_multi_gpu()` retorna resultados **na mesma ordem** que `tasks`;
- se `--multi-gpu` é passado com < 2 GPUs disponíveis, emitir aviso e continuar em single-GPU (não abortar);
- stdlib `multiprocessing` — sem novas dependências.

## Sequência mínima esperada

1. Criar `src/upscale_image/pipeline/multi_gpu.py` com `_gpu_worker` e `run_batch_multi_gpu`.
2. Adicionar `multi_gpu: bool = False` e `gpu_ids: list[int]` em `config/schema.py`.
3. Adicionar branch `multi_gpu` em `pipeline/batch.py`.
4. Adicionar `--multi-gpu` em `cli/main.py` com validação e aviso para < 2 GPUs.
5. Criar `tests/test_multi_gpu.py` com mocks de CUDA e multiprocessing.
6. Executar `pytest tests/test_multi_gpu.py -v`.
7. Executar `pytest tests/` e garantir regressão completa.

## Antes de finalizar

- executar `pytest tests/test_multi_gpu.py tests/test_batch.py -v`;
- confirmar que `pytest tests/` passa completamente;
- atualizar `prompt/control.yaml`.
