# Prompt do Passo 27

Execute o passo 27 do projeto: integrar o modelo SwinIR como runner alternativo.

## Leia antes de implementar

- `/home/edgar/dev/upscale_image/docs/parts/27-integrar-modelo-swinir.md`
- `/home/edgar/dev/upscale_image/docs/adr/0004-contrato-de-modelo-e-registry.md`
- `/home/edgar/dev/upscale_image/docs/adr/0007-stack-tecnologica-principal.md`
- `/home/edgar/dev/upscale_image/src/upscale_image/models/base.py`
- `/home/edgar/dev/upscale_image/src/upscale_image/models/realesrgan.py`
- `/home/edgar/dev/upscale_image/src/upscale_image/models/registry.py`

## Regras fixas

- registro de `SwinIRRunner` em `registry.py` deve ser **condicional** (`try/except ImportError`) — não quebrar quando `timm` não instalado;
- `SwinIRRunner` deve aplicar padding para múltiplos de `window_size=8` antes do forward e cortar o output de volta;
- não registrar `SwinIRRunner` incondicionalmente;
- o pipeline não muda — apenas um novo runner é adicionado via registro (ADR 0004).

## Sequência mínima esperada

1. Criar `requirements/swinir.txt` com `timm>=1.0.0` e atualizar `pyproject.toml`.
2. Criar `src/upscale_image/models/swinir_runner.py` com o contrato completo.
3. Adicionar registro condicional em `src/upscale_image/models/registry.py`.
4. Criar `tests/test_swinir_runner.py` usando mocks para `SwinIRNet` e pesos.
5. Executar `pytest tests/test_swinir_runner.py -v`.
6. Executar `pytest tests/` e garantir regressão completa.

## Antes de finalizar

- executar `pytest tests/test_swinir_runner.py tests/test_registry.py -v`;
- confirmar que `pytest tests/` passa completamente;
- atualizar `prompt/control.yaml`.
