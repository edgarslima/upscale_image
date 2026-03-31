# Prompt do Passo 24

Execute o passo 24 do projeto: aplicar quick wins de compilação de modelo, AMP automático e correção de tile_pad.

## Leia antes de implementar

- `/home/edgar/dev/upscale_image/docs/parts/24-fase1-quick-wins-compilacao-amp-tile.md`
- `/home/edgar/dev/upscale_image/docs/adr/0011-compilacao-de-modelo-e-amp-automatico.md`
- `/home/edgar/dev/upscale_image/docs/adr/0007-stack-tecnologica-principal.md`
- `/home/edgar/dev/upscale_image/src/upscale_image/config/schema.py`
- `/home/edgar/dev/upscale_image/src/upscale_image/models/realesrgan.py`
- `/home/edgar/dev/upscale_image/tests/test_config.py`
- `/home/edgar/dev/upscale_image/tests/test_realesrgan_runner.py`

## Regras fixas

- não alterar interfaces públicas de `SuperResolutionModel` (`load`, `upscale`, `unload`);
- aplicar `torch.compile` e `cudnn.benchmark` apenas quando `torch.cuda.is_available()`;
- usar `torch.autocast` no lugar de `tensor.half()` / `net.half()` — nunca coexistir com AMP manual;
- `tile_pad` default passa de `10` para `32` em `RuntimeConfig`;
- nenhuma nova dependência externa — tudo já está em `requirements/base.txt`.

## Sequência mínima esperada

1. Corrigir `tile_pad` default em `schema.py` e atualizar `tests/test_config.py`.
2. Adicionar `cudnn.benchmark` e `torch.compile` no método `load()` de `RealESRGANRunner`.
3. Substituir AMP manual por `torch.autocast` no método `upscale()`.
4. Reescrever `_upscale_tiled()` com acumulação ponderada usando janela de Hann; adicionar `_hann_window()`.
5. Remover asserções `isinstance(model._net, RRDBNet)` em `tests/test_realesrgan_runner.py`.
6. Criar `tests/test_quickwins.py` com cobertura dos novos comportamentos.
7. Executar `pytest tests/` e garantir regressão completa.

## Antes de finalizar

- executar `pytest tests/test_config.py tests/test_realesrgan_runner.py tests/test_quickwins.py -v`;
- confirmar que `pytest tests/` passa completamente;
- atualizar `prompt/control.yaml`.
