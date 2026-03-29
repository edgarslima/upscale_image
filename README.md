# upscale-image

CLI para super-resolução de imagens em lote. Aplica modelos de deep learning a coleções de imagens, mede qualidade e produz artefatos auditáveis por rodada (outputs, métricas, logs, manifesto JSON).

## Requisitos

- Python 3.12+
- pip
- GPU NVIDIA com CUDA (opcional, mas recomendado para Real-ESRGAN)

---

## Instalação rápida (nova máquina)

```bash
# 1. Clonar
git clone <url-do-repositorio>
cd upscale_image

# 2. Criar ambiente virtual isolado dentro da pasta
python3 -m venv .venv

# 3. Ativar
source .venv/bin/activate          # Linux / macOS
# .venv\Scripts\activate           # Windows

# 4. Instalar com tudo (inclui métricas e pytest)
pip install -r requirements/dev.txt
pip install -e .
```

Para desativar: `deactivate`

> `weights/` e `runs/` **não estão no repositório** (estão no `.gitignore`). Veja as seções abaixo.

---

## Perfis de instalação

| Perfil | Comando | Inclui |
|---|---|---|
| Só runtime | `pip install -r requirements/base.txt && pip install -e .` | CLI, modelos bicubic/mock |
| Com métricas | `pip install -r requirements/benchmark.txt && pip install -e ".[benchmark]"` | + PSNR, SSIM, LPIPS, NIQE |
| Desenvolvimento | `pip install -r requirements/dev.txt && pip install -e ".[dev]"` | + pytest |
| Real-ESRGAN | `pip install -r requirements/realesrgan-extra.txt && pip install -e ".[benchmark]"` | + basicsr, facexlib, gfpgan |

---

## Modelos disponíveis

| Nome | Tipo | Pesos |
|---|---|---|
| `bicubic` | Interpolação bicubic (PyTorch) | Não precisa |
| `mock` | Nearest-neighbour, apenas para testes | Não precisa |
| `realesrgan-x4` | Real-ESRGAN ×4 (deep learning) | Download manual |

### Baixar pesos do Real-ESRGAN

```bash
mkdir -p weights
wget -O weights/realesrgan-x4.pth <url-do-arquivo-pth>
```

---

## Uso

### Upscale básico

```bash
upscale-image upscale /fotos/originais --output /fotos/saida
```

### Com modelo e device explícitos

```bash
upscale-image upscale /fotos/originais \
  --output /fotos/saida \
  --model realesrgan-x4 \
  --scale 4 \
  --device cuda
```

### Com arquivo de configuração YAML

```bash
upscale-image upscale /fotos/originais --output /fotos/saida --config configs/minha_config.yaml
```

`configs/minha_config.yaml`:

```yaml
model:
  name: bicubic
  scale: 4
runtime:
  device: cpu
  precision: fp32
```

A precedência é: **flags CLI > arquivo YAML > defaults**.

### Com benchmark de qualidade

```bash
# Full-reference (precisa das imagens originais em alta resolução)
upscale-image upscale /fotos --output /saida --reference-dir /fotos-hd

# No-reference (NIQE, sem imagens de referência)
upscale-image upscale /fotos --output /saida --benchmark-nr
```

### Comparar rodadas

```bash
upscale-image compare runs/run_20260328_120000_bicubic_4x runs/run_20260328_130000_realesrgan-x4_4x

# Salvar JSON
upscale-image compare runs/run_A runs/run_B --output comparacao.json
```

### Relatório HTML

```bash
upscale-image report runs/run_A runs/run_B --output relatorio.html
```

---

## Artefatos de cada rodada

Cada execução cria automaticamente um diretório em `runs/`:

```
runs/run_20260328_120000_bicubic_4x/
├── outputs/              # Imagens geradas (.png)
├── metrics/
│   ├── summary.json          # Médias PSNR/SSIM/LPIPS (com --reference-dir)
│   ├── per_image.csv         # Métricas por imagem
│   ├── niqe_summary.json     # Média NIQE (com --benchmark-nr)
│   └── niqe_per_image.csv
├── manifest.json         # Modelo, timing, status — fonte de verdade
├── effective_config.yaml # Configuração exata usada (reprodutibilidade)
└── logs.txt              # Log completo
```

---

## Estrutura do projeto

```
src/upscale_image/
  cli/        # Comandos Typer: upscale, compare, report
  config/     # Merge de configuração: CLI > YAML > defaults
  io/         # Descoberta e validação de imagens
  models/     # Contrato ABC, registry, runners (bicubic, mock, realesrgan)
  pipeline/   # Loop batch, run context, manifesto
  metrics/    # PSNR/SSIM/LPIPS (full-reference) e NIQE (no-reference)
  reports/    # Comparação entre runs e relatório HTML
requirements/ # Dependências por perfil (base, benchmark, dev, realesrgan-extra)
configs/      # Exemplos de configuração YAML
weights/      # Pesos dos modelos (não versionado, download manual)
runs/         # Saídas das execuções (não versionado, gerado em tempo de execução)
tests/        # 314 testes automatizados
```

---

## Testes

```bash
pytest                        # suite completa
pytest tests/test_batch.py    # arquivo específico
pytest -k "bicubic"           # filtrar por nome
pytest -v                     # saída detalhada
```

---

## Formatos suportados

`.jpg` `.jpeg` `.png` `.bmp` `.tiff` `.tif` `.webp`

---

## Códigos de saída

| Código | Significado |
|---|---|
| `0` | Sucesso total |
| `1` | Erro estrutural (config inválida, modelo não encontrado, pesos ausentes) |
| `2` | Concluído com falhas parciais em algumas imagens |
