# Passo 26: Fase 3 — Batch Inference Real (batch_size > 1)

## Objetivo

Processar N imagens em um único forward pass GPU, multiplicando o throughput por fator
próximo a N (com retornos decrescentes por limitação de VRAM). Ganho esperado:
**2–4× adicional** sobre o throughput da Fase 2, dependendo do `batch_size` e do hardware.

## ADRs relacionadas

- [0004-contrato-de-modelo-e-registry.md](../adr/0004-contrato-de-modelo-e-registry.md) — `upscale_batch()` é adicionado ao contrato com implementação default que mantém compatibilidade total
- [0012-batch-inference.md](../adr/0012-batch-inference.md) — decisão de estratégia de batching, agrupamento por tamanho e gestão dinâmica de VRAM
- [0005-pipeline-deterministico-e-estrategia-de-erros.md](../adr/0005-pipeline-deterministico-e-estrategia-de-erros.md) — estratégia de erro por item se aplica ao nível do batch também

## Problema a resolver

O pipeline atual faz um forward pass por imagem. A GPU processa imagens uma a uma,
desperdiçando capacidade de paralelismo do hardware. Com `batch_size=4`, o throughput
pode triplicar sem aumentar o consumo de VRAM proporcionalmente (kernels CUDA ganham
eficiência com batches maiores).

**Desafio**: imagens têm tamanhos variados. Solução: padding para o maior tamanho do
batch, forward pass, corte dos resultados de volta ao tamanho original por imagem.

## Dependências

- Passo 25 concluído (pipeline assíncrono operacional — `async_worker.py` integrado)
- `torch.nn.functional.pad` e `numpy` já disponíveis (sem novas dependências)

## Entregáveis

1. Método `upscale_batch()` adicionado à interface `SuperResolutionModel` em `models/base.py` com implementação default (loop sobre `upscale()`)
2. Implementação real de `upscale_batch()` em `models/realesrgan.py` com padding, forward pass único e corte
3. Função `group_tasks_by_size()` em `pipeline/batch.py` ou módulo auxiliar para agrupar tasks por dimensão similar
4. Função `estimate_safe_batch_size()` para auto-detect baseado em VRAM disponível
5. `pipeline/async_worker.py` atualizado: thread de inferência acumula batch e chama `upscale_batch()`
6. `config/schema.py` atualizado: `batch_size: int = 1` (0 = auto-detect)
7. `cli/main.py` atualizado: flag `--batch-size`
8. `tests/test_batch_inference.py` cobrindo a nova lógica

## Escopo obrigatório

### 6.1 Adicionar `upscale_batch()` ao contrato base

**Arquivo**: `src/upscale_image/models/base.py`

Adicionar método com implementação default ao `SuperResolutionModel`:

```python
def upscale_batch(
    self,
    images: list[np.ndarray],
    config: "AppConfig",
) -> list[np.ndarray]:
    """Upscale um lote de imagens. Default: loop serial sobre upscale().

    Runners que suportam batch real devem sobrescrever este método.
    Compatibilidade total: qualquer runner existente continua funcionando
    sem alteração.
    """
    return [self.upscale(img, config) for img in images]
```

### 6.2 Implementação real em `RealESRGANRunner`

**Arquivo**: `src/upscale_image/models/realesrgan.py`

Sobrescrever `upscale_batch()`:

```python
def upscale_batch(self, images: list[np.ndarray], config: AppConfig) -> list[np.ndarray]:
    device = self._resolve_device(config.runtime.device)
    net    = self._net.to(device).float()
    use_amp = (config.runtime.precision == "fp16") and (device.type == "cuda")

    # Converter todas as imagens para tensors FP32
    tensors = []
    original_sizes = []
    for img in images:
        h, w = img.shape[:2]
        original_sizes.append((h, w))
        rgb = img[:, :, ::-1].copy()
        t = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0)
        tensors.append(t)

    # Paddar para o maior tamanho do batch
    max_h = max(t.shape[1] for t in tensors)
    max_w = max(t.shape[2] for t in tensors)
    padded = []
    for t in tensors:
        pad_h = max_h - t.shape[1]
        pad_w = max_w - t.shape[2]
        import torch.nn.functional as F
        padded.append(F.pad(t, (0, pad_w, 0, pad_h), mode="reflect"))

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
        results.append(out_np[:, :, ::-1].copy())
    return results
```

**Nota**: `upscale_batch()` não suporta tiling por enquanto. Imagens maiores que a VRAM
disponível devem ser processadas individualmente via `upscale()`. Implementar fallback:
se a imagem for maior que `config.runtime.tile_size` (quando tile ativo), redirecionar
para `upscale()`.

### 6.3 Agrupamento de tasks por tamanho

**Arquivo**: `src/upscale_image/pipeline/batch.py` (ou `pipeline/grouping.py`)

```python
def group_tasks_by_size(
    tasks: list[ImageTask],
    batch_size: int,
    size_tolerance: float = 0.2,
) -> list[list[ImageTask]]:
    """Agrupa tasks em lotes de tamanho similar para minimizar padding."""
```

Estratégia:
1. Ler dimensões de cada imagem (apenas header, sem carregar pixels)
2. Ordenar por área (maior → menor)
3. Agrupar: acumular até `batch_size` tasks com variação de área ≤ `size_tolerance` (20%)
4. Retornar lista de grupos — cada grupo é processado em um forward pass

### 6.4 Auto-detect de `batch_size` por VRAM

**Arquivo**: `src/upscale_image/pipeline/batch.py`

```python
def estimate_safe_batch_size(
    sample_image: np.ndarray,
    model: SuperResolutionModel,
    config: AppConfig,
    safety_factor: float = 0.7,
) -> int:
    """Estima batch_size máximo seguro baseado na VRAM disponível."""
    if not torch.cuda.is_available():
        return 1
    total_vram = torch.cuda.get_device_properties(0).total_memory
    free_vram  = total_vram - torch.cuda.memory_allocated(0)
    usable     = int(free_vram * safety_factor)
    h, w, c = sample_image.shape
    # RRDBNet x4 usa ~8× o tamanho do tensor de input em VRAM durante forward
    bytes_per_image = h * w * c * 4 * 8 * (config.model.scale ** 2)
    return max(1, usable // bytes_per_image)
```

Quando `config.runtime.batch_size == 0`, chamar `estimate_safe_batch_size()` com a
primeira imagem da lista antes de iniciar o agrupamento.

### 6.5 Integração no pipeline assíncrono

**Arquivo**: `src/upscale_image/pipeline/async_worker.py`

Quando `config.runtime.batch_size > 1`, a thread de inferência acumula itens de
`read_queue` até atingir `batch_size` ou receber `_STOP`, então chama
`model.upscale_batch()` com o grupo.

Erros em `upscale_batch()`:
- Se o batch inteiro falha (exceção geral), converter cada item do batch em `ItemResult(status="failed")`
- Erros por imagem individual dentro do batch não são distinguíveis neste nível — o item inteiro é marcado como falho

### 6.6 Novos campos em `RuntimeConfig` e CLI

**`config/schema.py`**:
```python
batch_size: int = 1   # 0 = auto-detect por VRAM
```

**`cli/main.py`**:
```python
batch_size: int = typer.Option(1, "--batch-size", "-b",
    help="Images per GPU forward pass. 0 = auto-detect by VRAM.")
```

## Fora de escopo

- Tiling com batch inference (combinação Fase 1 + Fase 3) — fora deste passo
- Multi-GPU (é o Passo 29)
- Novos runners (SwinIR é o Passo 27)

## Novos testes a criar

**Arquivo**: `tests/test_batch_inference.py`

Cobrir:
- `upscale_batch()` default na interface base chama `upscale()` N vezes e retorna N resultados
- `MockSuperResolutionModel.upscale_batch()` funciona (herda default)
- `RealESRGANRunner.upscale_batch()` retorna lista com N elementos de shapes corretos (mock CUDA)
- Padding: batch com imagens de tamanhos diferentes retorna outputs com shapes individuais corretos (não padded)
- `group_tasks_by_size()` agrupa corretamente por área com tolerância de 20%
- `group_tasks_by_size()` com `batch_size=1` retorna N grupos de 1 elemento cada
- `estimate_safe_batch_size()` retorna 1 quando CUDA não disponível (sem mock)
- `run_batch()` com `batch_size=1` não sofre regressão
- Regressão completa: todos os testes anteriores passando

## Sequência sugerida para implementação

1. Ler `src/upscale_image/models/base.py` e `realesrgan.py` antes de editar.
2. Adicionar `upscale_batch()` ao contrato em `base.py` com implementação default.
3. Implementar `upscale_batch()` real em `realesrgan.py`.
4. Implementar `group_tasks_by_size()` e `estimate_safe_batch_size()`.
5. Integrar batch no `async_worker.py`.
6. Atualizar `schema.py` e `cli/main.py`.
7. Criar `tests/test_batch_inference.py`.
8. Executar `pytest tests/` para validar regressão completa.

## Critérios de aceite

- `SuperResolutionModel.upscale_batch()` existe com implementação default
- `RealESRGANRunner.upscale_batch()` retorna outputs com shapes `(orig_h * scale, orig_w * scale, 3)` para cada imagem
- `group_tasks_by_size()` agrupa corretamente; `estimate_safe_batch_size()` retorna ≥ 1
- `config.runtime.batch_size` existe, default `1`, aceita `0` para auto-detect
- Flag `--batch-size` disponível na CLI
- `pytest tests/` passa completamente

## Como testar

```bash
pytest tests/test_batch_inference.py -v
pytest tests/test_async_worker.py -v   # regressão da Fase 2
pytest tests/ -q                        # regressão completa
```

## Armadilhas a evitar

- Não retornar outputs padded — cortar de volta ao tamanho `orig_h * scale, orig_w * scale` de cada imagem
- Não assumir que todas as imagens do batch têm o mesmo tamanho no output
- Não usar `upscale_batch()` diretamente no modo serial (apenas no modo assíncrono ou quando explicitamente solicitado)
- Não quebrar a implementação default do `upscale_batch()` na interface base — deve ser retrocompatível com runners existentes
