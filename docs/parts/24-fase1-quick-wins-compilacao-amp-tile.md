# Passo 24: Fase 1 — Quick Wins: Compilação de Modelo, AMP Automático e Tile Pad

## Objetivo

Aplicar um conjunto de otimizações de baixo risco e alto impacto ao pipeline existente,
sem alterar interfaces públicas. Obter ganho de **2–3× em throughput** apenas com
mudanças internas ao runner `RealESRGANRunner` e à configuração padrão.

## ADRs relacionadas

- [0007-stack-tecnologica-principal.md](../adr/0007-stack-tecnologica-principal.md) — stack atualizada para incluir `torch.compile` e `torch.autocast`
- [0011-compilacao-de-modelo-e-amp-automatico.md](../adr/0011-compilacao-de-modelo-e-amp-automatico.md) — decisão de compilar o modelo e usar AMP automático em CUDA

## Problema a resolver

O runner atual apresenta quatro ineficiências documentadas em `docs/plano_otimizacao.md`:

| Problema | Local | Impacto |
|---|---|---|
| `tile_pad=10` causa artefatos visuais | `config/schema.py:23` | Qualidade visual |
| cuDNN autotuner desabilitado | `models/realesrgan.py` — método `load()` | 5–15% throughput |
| Modelo não compilado | `models/realesrgan.py` — método `load()` | 20–40% throughput |
| AMP manual frágil (`tensor.half()`) | `models/realesrgan.py` — método `upscale()` | 30–50% em CUDA, risco de `nan` |
| Tiling sem blend suavizado | `models/realesrgan.py` — método `_upscale_tiled()` | Bordas visíveis entre tiles |

## Dependências

- Passos 1–18 concluídos (pipeline base operacional)
- `torch==2.11.0` instalado (já presente em `requirements/base.txt`) — todas as APIs usadas aqui (`torch.compile`, `torch.autocast`, `torch.backends.cudnn`) estão disponíveis nessa versão

## Entregáveis

1. `tile_pad` default corrigido de `10` → `32` em `config/schema.py`
2. `torch.backends.cudnn.benchmark = True` habilitado em `load()` quando CUDA disponível
3. `torch.compile(net, mode="reduce-overhead")` aplicado em `load()` quando CUDA disponível
4. `torch.autocast` substituindo conversão manual `tensor.half()` em `upscale()`
5. Gaussian feathering (janela de Hann) em `_upscale_tiled()` para eliminar bordas entre tiles
6. ADR 0011 criada ✓ (já presente em `docs/adr/`)
7. ADR 0007 atualizada ✓ (já atualizada para incluir `torch.compile` e `torch.autocast`)
8. Testes existentes atualizados para refletir as mudanças
9. Novos testes cobrindo os comportamentos adicionados

## Escopo obrigatório

### 4.1 Corrigir `tile_pad` default

**Arquivo**: `src/upscale_image/config/schema.py`

```python
# Antes:
tile_pad: int = 10

# Depois:
tile_pad: int = 32
```

O padding mínimo para cobrir o campo receptivo do RRDBNet x4 é 32. O valor 10 produz
artefatos visíveis nas bordas de tiles. Atualizar também o teste em `tests/test_config.py`
que verifica o valor default de `tile_pad`.

### 4.2 Habilitar `cudnn.benchmark` no `load()`

**Arquivo**: `src/upscale_image/models/realesrgan.py` — método `load()`

Inserir após `net.eval()`, antes de atribuir `self._net`:

```python
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
```

`cudnn.benchmark=True` aciona o autotuner de algoritmos de convolução. Custo: primeiras
chamadas são ~2× mais lentas enquanto o autotuner executa. Benefício: todas as chamadas
subsequentes usam o algoritmo ótimo para os tamanhos de tensor encontrados.

**Nota**: é um estado global do processo. Se múltiplos runners forem instanciados
(Fase 5), a flag já estará ativa desde a inicialização do primeiro.

### 4.3 Substituir AMP manual por `torch.autocast`

**Arquivo**: `src/upscale_image/models/realesrgan.py` — método `upscale()`

A conversão manual `net.half()` / `tensor.half()` é frágil: qualquer operação que
precise de FP32 internamente causa `nan` silencioso. O `torch.autocast` gerencia
automaticamente quais ops usam FP16 (convoluções, matmuls) e quais permanecem em FP32.

Estrutura esperada do método `upscale()`:

```python
device = self._resolve_device(config.runtime.device)
net = self._net.to(device)
net = net.float()  # modelo sempre em FP32 em memória

# Tensor de entrada sempre em FP32
img_rgb = image[:, :, ::-1].copy()
tensor = (
    torch.from_numpy(img_rgb)
    .permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)
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
    output.squeeze(0).float().clamp(0.0, 1.0)
    .mul(255.0).round().byte()
    .permute(1, 2, 0).cpu().numpy()
)
return out_np[:, :, ::-1].copy()
```

### 4.4 Aplicar `torch.compile` no `load()`

**Arquivo**: `src/upscale_image/models/realesrgan.py` — método `load()`

Inserir após `cudnn.benchmark`, antes de atribuir `self._net`:

```python
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    net = torch.compile(net, mode="reduce-overhead")
```

`torch.compile` transforma o modelo em `OptimizedModule`. Isso é compatível com a
interface `nn.Module` — `.to(device)`, `.float()` funcionam normalmente. A compilação
é lazy: ocorre na primeira chamada de `forward`, não no `load()`.

**Impacto em testes**: remover qualquer asserção `isinstance(model._net, RRDBNet)`
em `tests/test_realesrgan_runner.py` — é detalhe de implementação exposto
incorretamente. Substituir por testes de comportamento (output shape, dtype).

### 4.5 Gaussian feathering (janela de Hann) no tiling

**Arquivo**: `src/upscale_image/models/realesrgan.py` — método `_upscale_tiled()`

O método atual faz corte duro entre tiles. Mesmo com `tile_pad=32`, gradientes fortes
podem produzir bordas. A solução definitiva é usar acumulação ponderada com janela de Hann.

Estrutura esperada:

```python
def _upscale_tiled(self, tensor, net, config):
    tile_size = config.runtime.tile_size
    tile_pad  = config.runtime.tile_pad
    scale     = self._scale

    _, channel, height, width = tensor.shape
    output_sum = tensor.new_zeros(1, channel, height * scale, width * scale)
    weight_sum = tensor.new_zeros(1, 1,       height * scale, width * scale)

    tiles_x = math.ceil(width  / tile_size)
    tiles_y = math.ceil(height / tile_size)

    for iy in range(tiles_y):
        for ix in range(tiles_x):
            # Calcular coordenadas de entrada com padding
            in_x1, in_x2 = ix * tile_size, min((ix + 1) * tile_size, width)
            in_y1, in_y2 = iy * tile_size, min((iy + 1) * tile_size, height)
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

            h_out, w_out = cy2 - cy1, cx2 - cx1
            mask = self._hann_window(h_out, w_out, device=tensor.device)

            out_y1, out_y2 = in_y1 * scale, in_y2 * scale
            out_x1, out_x2 = in_x1 * scale, in_x2 * scale
            output_sum[:, :, out_y1:out_y2, out_x1:out_x2] += valid * mask
            weight_sum[:,  :, out_y1:out_y2, out_x1:out_x2] += mask

    return output_sum / weight_sum.clamp(min=1e-6)

@staticmethod
def _hann_window(h: int, w: int, device: torch.device) -> torch.Tensor:
    """Janela de Hann 2D: máximo 1 no centro, suave até 0 nas bordas."""
    hann_h = torch.hann_window(h, periodic=False, device=device)
    hann_w = torch.hann_window(w, periodic=False, device=device)
    return (hann_h.unsqueeze(1) * hann_w.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
```

O módulo `math` já deve ser importado ou adicionar `import math` no topo do arquivo.

## Fora de escopo

- Alterações no pipeline de batch ou na CLI
- Qualquer mudança que afete a interface pública (`load`, `upscale`, `unload`)
- Adição de novas dependências (tudo já está em `requirements/base.txt`)
- Implementação das Fases 2–5 (são passos separados)

## Impacto em testes existentes

| Arquivo | Motivo da atualização |
|---|---|
| `tests/test_config.py` | `tile_pad` default mudou de `10` → `32` |
| `tests/test_realesrgan_runner.py` | Remover asserção `isinstance(._net, RRDBNet)` — quebra após `torch.compile` |

## Novos testes a criar

**Arquivo**: `tests/test_quickwins.py`

Cobrir:
- `tile_pad` default é 32 após a mudança no schema
- `cudnn.benchmark` é ativado quando CUDA disponível (mock `torch.cuda.is_available`)
- `upscale()` com `precision="fp16"` e `device="cuda"` usa `autocast` corretamente (mock)
- `upscale()` com `precision="fp32"` não usa `autocast` (mock)
- `_hann_window` retorna tensor com shape `(1, 1, h, w)`, dtype float, valores em `[0, 1]`
- Tiling com janela de Hann produz output com shape correto (sem CUDA — usar mock runner)
- Regressão completa: todos os testes anteriores continuam passando

## Sequência sugerida para implementação

1. Ler `src/upscale_image/config/schema.py` e `src/upscale_image/models/realesrgan.py` antes de editar.
2. Corrigir `tile_pad` default e atualizar `tests/test_config.py`.
3. Adicionar `cudnn.benchmark` e `torch.compile` no `load()`.
4. Substituir AMP manual por `torch.autocast` em `upscale()`.
5. Implementar `_hann_window` e reescrever `_upscale_tiled()` com acumulação ponderada.
6. Executar `pytest tests/test_config.py tests/test_realesrgan_runner.py` e corrigir falhas.
7. Criar `tests/test_quickwins.py` com cobertura das novas funcionalidades.
8. Executar `pytest` completo para validar regressão.

## Critérios de aceite

- `tile_pad` default é `32` em `RuntimeConfig`
- `torch.backends.cudnn.benchmark` é `True` após `load()` quando `torch.cuda.is_available()` retorna `True`
- `upscale()` usa `torch.autocast` quando `precision="fp16"` e `device="cuda"`; não usa em FP32 ou CPU
- `_upscale_tiled()` usa acumulação ponderada com janela de Hann; o método `_hann_window` existe e retorna shape correto
- Nenhum teste pré-existente quebra (ajustar `test_config.py` e `test_realesrgan_runner.py` onde necessário)
- `pytest tests/` passa completamente

## Como testar

```bash
# Executar testes afetados
pytest tests/test_config.py tests/test_realesrgan_runner.py -v

# Executar novos testes
pytest tests/test_quickwins.py -v

# Regressão completa
pytest tests/ -q
```

## Armadilhas a evitar

- Não testar `isinstance(model._net, RRDBNet)` após `torch.compile` — o tipo muda para `OptimizedModule`
- Não aplicar `torch.compile` quando CUDA não está disponível — pode falhar ou ser inútil em CPU
- Não remover o `with torch.inference_mode()` externo ao aplicar `autocast` — ambos devem coexistir
- Não atualizar o `tile_pad` default sem atualizar o teste correspondente em `test_config.py`
