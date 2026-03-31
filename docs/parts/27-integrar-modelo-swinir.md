# Passo 27: Integrar Modelo SwinIR

## Objetivo

Adicionar o modelo **SwinIR Real-SR (Large, x4)** como segunda opção de runner para
super-resolução, registrado no `ModelRegistry` de forma condicional (apenas quando o
pacote `swinir` estiver instalado). O pipeline não muda — apenas um novo runner é
adicionado via registro, conforme ADR 0004.

SwinIR produz **menos alucinações em texto e PDF**, bordas mais precisas e menos
artefatos em regiões de baixa textura — especialmente relevante para o fluxo PDF
implementado nos passos 21–23.

## ADRs relacionadas

- [0004-contrato-de-modelo-e-registry.md](../adr/0004-contrato-de-modelo-e-registry.md) — novo modelo é adicionado por registro, sem alterar o pipeline
- [0007-stack-tecnologica-principal.md](../adr/0007-stack-tecnologica-principal.md) — `timm` adicionado como dependência opcional para SwinIR

## Dependências

- Passo 26 concluído (`upscale_batch()` no contrato base — SwinIR herdará o default)
- `timm>=1.0.0` disponível via `requirements/swinir.txt` (a criar neste passo)

## Entregáveis

1. `src/upscale_image/models/swinir_runner.py` — runner completo implementando `SuperResolutionModel`
2. `requirements/swinir.txt` — dependência `timm` com pin de versão
3. `src/upscale_image/models/registry.py` atualizado com registro condicional de `SwinIRRunner`
4. `pyproject.toml` atualizado: opcional extra `swinir = ["timm>=1.0.0"]`
5. `tests/test_swinir_runner.py` cobrindo contrato e comportamento esperado

## Escopo obrigatório

### 9.1 Dependência

**Arquivo a criar**: `requirements/swinir.txt`

```
-r base.txt
timm>=1.0.0
```

**`pyproject.toml`** — adicionar optional extra:

```toml
[project.optional-dependencies]
swinir = ["timm>=1.0.0"]
```

### 9.2 Runner `SwinIRRunner`

**Arquivo a criar**: `src/upscale_image/models/swinir_runner.py`

O runner deve implementar completamente o contrato `SuperResolutionModel`:
- `name` → `"swinir-x4"`
- `scale` → fator de upscale configurado (padrão `4`)
- `is_loaded` → `True` após `load()` bem-sucedido
- `load()` → carrega pesos de `weights_path`, aplica `torch.compile` quando CUDA disponível
- `upscale()` → inferência com `torch.autocast`, padding para múltiplos de `window_size=8`, corte do resultado
- `unload()` → libera modelo e chama `torch.cuda.empty_cache()`

**Particularidades do SwinIR**:

- Requer que `H` e `W` do tensor de entrada sejam **múltiplos de `window_size=8`**. Aplicar
  padding reflect antes do forward e cortar o output de volta ao tamanho original × scale.
- Arquitetura configurada para Real-SR Large x4:
  ```python
  SwinIR(
      upscale=4, in_chans=3, img_size=64, window_size=8, img_range=1.0,
      depths=[6,6,6,6,6,6,6,6,6], embed_dim=240,
      num_heads=[8,8,8,8,8,8,8,8,8], mlp_ratio=2,
      upsampler="nearest+conv", resi_connection="3conv",
  )
  ```
- Pesos oficiais carregados com `torch.load(..., weights_only=True)`. Se o state dict
  contém chave `"params_ema"` ou `"params"`, extrair antes de `load_state_dict()`.
- `torch.compile(net, mode="reduce-overhead")` quando `torch.cuda.is_available()`.

**Tratamento de `ImportError`**:

O runner deve falhar com `ImportError` claro em `load()` se `swinir` não estiver instalado:

```python
try:
    from swinir import SwinIR as SwinIRNet
except ImportError:
    raise ImportError(
        "SwinIR não instalado. Instale com: "
        "pip install -r requirements/swinir.txt"
    )
```

### 9.3 Registro condicional no registry

**Arquivo**: `src/upscale_image/models/registry.py`

Adicionar registro condicional ao final do arquivo (após os registros existentes):

```python
try:
    from upscale_image.models.swinir_runner import SwinIRRunner
    _registry.register("swinir-x4", lambda: SwinIRRunner(scale=4))
except ImportError:
    pass  # SwinIR não instalado — runner não disponível
```

O `try/except ImportError` garante que o registry continua funcionando normalmente
quando `timm` ou `swinir` não estão instalados.

### 9.4 Pesos esperados

Os pesos devem ser colocados em `weights/swinir-x4.pth`. O runner não baixa pesos
automaticamente — falha com `FileNotFoundError` claro se o arquivo não existir:

```python
if not Path(self._weights_path).exists():
    raise FileNotFoundError(
        f"Pesos SwinIR não encontrados: {self._weights_path}\n"
        "Baixe de: https://github.com/JingyunLiang/SwinIR/releases"
    )
```

## Fora de escopo

- Suporte a outros tamanhos ou variantes do SwinIR (apenas Real-SR Large x4)
- Download automático de pesos
- Tiling específico para SwinIR (pode reusar `_upscale_tiled()` do RealESRGAN em versão futura)
- Implementação de `upscale_batch()` específica para SwinIR (herda o default do contrato base)

## Novos testes a criar

**Arquivo**: `tests/test_swinir_runner.py`

Cobrir:
- `SwinIRRunner` implementa `SuperResolutionModel` (ABC)
- `name` retorna `"swinir-x4"`, `scale` retorna `4`, `is_loaded` começa `False`
- `load()` falha com `FileNotFoundError` quando `weights_path` não existe
- `load()` falha com `ImportError` claro quando `swinir` não está instalado (mock `ImportError`)
- `upscale()` falha com `RuntimeError` quando chamado sem `load()`
- `upscale()` com imagem de tamanho não múltiplo de 8 retorna output com shape `(H * scale, W * scale, 3)` (mock do `_net`)
- Padding reflect é aplicado e o output é cortado corretamente (mock do `_net` que retorna tensor de tamanho padded)
- `unload()` limpa o modelo e `is_loaded` retorna `False`
- Registro condicional: `"swinir-x4"` está em `registry.available()` quando `SwinIRRunner` pode ser importado

## Sequência sugerida para implementação

1. Criar `requirements/swinir.txt` e atualizar `pyproject.toml`.
2. Criar `src/upscale_image/models/swinir_runner.py` com estrutura completa.
3. Adicionar registro condicional em `registry.py`.
4. Criar `tests/test_swinir_runner.py` usando mocks para `SwinIRNet` e pesos.
5. Executar `pytest tests/test_swinir_runner.py -v`.
6. Executar `pytest tests/` para regressão completa.

## Critérios de aceite

- `SwinIRRunner` implementa o contrato `SuperResolutionModel` completo
- `upscale()` aplica padding para múltiplos de 8 e corta o output corretamente
- `load()` emite erros claros para peso ausente (`FileNotFoundError`) e pacote ausente (`ImportError`)
- Registry registra `"swinir-x4"` condicionalmente (sem quebrar quando `timm` não instalado)
- `requirements/swinir.txt` existe com `timm>=1.0.0`
- `pytest tests/` passa completamente

## Como testar

```bash
pytest tests/test_swinir_runner.py -v
pytest tests/test_registry.py -v     # regressão do registry
pytest tests/ -q                      # regressão completa
```

## Armadilhas a evitar

- Não registrar `SwinIRRunner` incondicionalmente — a importação falharia em ambientes sem `timm`
- Não esquecer o padding para múltiplos de `window_size=8` — o SwinIR falha silenciosamente sem ele
- Não aplicar `torch.compile` em CPU — apenas quando `torch.cuda.is_available()`
- Não expor o tipo interno do modelo compilado em testes — usar testes de comportamento
