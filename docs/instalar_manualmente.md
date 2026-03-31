# Instalação Manual — Dependências de Performance

> Gerado em: 2026-03-30
> Referência: `docs/plano_otimizacao.md`

---

## Ambiente detectado

| Item | Valor |
|---|---|
| GPU | NVIDIA GeForce RTX 3070 (8 GB VRAM, 1 GPU) |
| Driver NVIDIA | 580.126.09 (suporte até CUDA 13.0) |
| CUDA toolkit (sistema/apt) | 12.0 |
| PyTorch | 2.11.0+cu130 |
| OS | Ubuntu 24.04 LTS |
| Kernel | 6.17.0-19-generic |
| timm (SwinIR) | 1.0.26 ✅ já instalado |

---

## O que falta e por que importa

| Dependência | Fase | Impacto | Requer sudo |
|---|---|---|---|
| **cuDNN 9** | 1–4 | Habilita `cudnn.benchmark` e é pré-requisito do TensorRT | ✅ Sim |
| **TensorRT 10** | 4 | Motor de inferência JIT para GPU NVIDIA — 2–4× mais rápido que PyTorch eager | ✅ Sim |
| **torch-tensorrt 2.11.0** | 4 | Integração Python do TensorRT com PyTorch | Não (pip) |
| **onnxruntime-gpu 1.20.0** | 4 | Backend alternativo ONNX com CUDA Execution Provider | Não (pip) |
| **2ª GPU NVIDIA** | 5 | Multi-GPU requer ≥ 2 GPUs — hardware atual tem apenas 1 | Hardware |

---

## Passo 1 — Adicionar o repositório NVIDIA (único sudo add-apt-repository)

```bash
# Baixar e instalar o keyring do NVIDIA Developer
wget -qO /tmp/cuda-keyring.deb \
  https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb

sudo dpkg -i /tmp/cuda-keyring.deb
sudo apt-get update
```

> Este repositório fornece cuDNN, TensorRT e NCCL via apt.
> Só precisa ser feito uma vez.

---

## Passo 2 — Instalar cuDNN 9 (CUDA 12)

cuDNN é a biblioteca de redes neurais profundas da NVIDIA. O código já chama
`torch.backends.cudnn.benchmark = True` (em `models/realesrgan.py`), que só
tem efeito quando cuDNN está disponível. Sem cuDNN, o autotuner de convolução
não ativa e não há ganho de performance nesta chamada.

```bash
# Bibliotecas de runtime do cuDNN (CUDA 12.x — compatível com a toolkit do sistema)
sudo apt-get install -y libcudnn9-cuda-12

# Cabeçalhos de desenvolvimento (necessários para compilar TensorRT plugins)
sudo apt-get install -y libcudnn9-dev-cuda-12

# Verificar instalação
dpkg -l | grep cudnn
# Esperado: libcudnn9-cuda-12   9.x.x.x-1
```

> **Por que CUDA 12 e não 13?** O toolkit de sistema (`nvcc`, `libcuda`) está
> em versão 12.0. O PyTorch usa CUDA 13.0 via runtime embutido, mas cuDNN
> interopera com ambos. Instalar o pacote `cuda-12` é a opção estável e
> disponível nos repositórios oficiais em abril de 2026.

---

## Passo 3 — Instalar TensorRT 10

TensorRT é o compilador/motor de inferência da NVIDIA. Habilita o runner
`TensorRTRunner` (Fase 4) que compila o modelo Real-ESRGAN em um engine
otimizado para a GPU específica, reduzindo latência em 2–4×.

```bash
# Instalar o TensorRT 10.x e suas dependências
sudo apt-get install -y tensorrt

# Instalar Python bindings do TensorRT (necessário para scripts de exportação)
sudo apt-get install -y python3-libnvinfer-dev

# Verificar instalação
dpkg -l | grep tensorrt
python3 -c "import tensorrt as trt; print('TensorRT:', trt.__version__)"
```

> **Espaço em disco**: TensorRT ocupa ~2–3 GB após instalação completa.

---

## Passo 4 — Instalar torch-tensorrt (sem sudo, dentro do venv)

torch-tensorrt é a ponte Python entre PyTorch e TensorRT. Permite compilar
modelos PyTorch para engines TRT e usa a interface `torch.compile`.

```bash
# Ativar o ambiente virtual do projeto
source .venv/bin/activate

# Instalar (versão alinhada com torch==2.11.0)
pip install torch-tensorrt==2.11.0

# Verificar
python -c "import torch_tensorrt; print('torch-tensorrt OK')"
```

> **Pré-requisito**: o Passo 3 (TensorRT 10 no sistema) deve estar concluído
> antes deste passo. `torch-tensorrt` usa as bibliotecas de sistema do TRT.

---

## Passo 5 — Instalar onnxruntime-gpu (sem sudo, dentro do venv)

ONNX Runtime com CUDA Execution Provider é o backend alternativo para rodar
o modelo via formato ONNX. Útil para comparar performance com TensorRT, ou
para ambientes sem suporte completo ao TRT.

```bash
# Ativar o ambiente virtual do projeto
source .venv/bin/activate

# Instalar (CUDA 12 EP — compatível com a toolkit de sistema)
pip install onnxruntime-gpu==1.20.0

# Verificar
python -c "
import onnxruntime as ort
print('ORT version:', ort.__version__)
print('Providers:', ort.get_available_providers())
"
# Esperado: 'CUDAExecutionProvider' na lista de providers
```

---

## Passo 6 — Exportar os modelos (pós-instalação)

Após instalar TensorRT e torch-tensorrt, os runners da Fase 4 precisam que
os modelos sejam exportados offline. Esse processo é específico por GPU —
o engine gerado não funciona em hardware diferente.

### 6a. Exportar para ONNX

```bash
source .venv/bin/activate

python scripts/export_onnx.py \
    --weights weights/realesrgan-x4.pth \
    --output  weights/realesrgan-x4.onnx
```

### 6b. Exportar para TensorRT (FP16 — recomendado para RTX 3070)

```bash
source .venv/bin/activate

python scripts/export_tensorrt.py \
    --weights    weights/realesrgan-x4.pth \
    --output     weights/realesrgan-x4-trt-fp16.ep \
    --precision  fp16 \
    --min-size   64 \
    --opt-size   512 \
    --max-size   2048
```

> A exportação TRT leva 2–10 minutos na primeira vez (compilação de kernels).
> O arquivo `.ep` gerado é específico para a RTX 3070 com a versão do TRT
> instalada. Deve ser regenerado ao atualizar TRT ou trocar de GPU.

---

## Passo 7 — Usar os novos backends na CLI

```bash
source .venv/bin/activate

# Usar runner TensorRT FP16 (máxima performance)
upscale-image upscale imagens/ --output runs/ \
    --model realesrgan-x4-trt-fp16 \
    --device cuda

# Usar runner ONNX com CUDA EP
upscale-image upscale imagens/ --output runs/ \
    --model realesrgan-x4-onnx-cuda \
    --device cuda

# Usar runner ONNX com CPU EP
upscale-image upscale imagens/ --output runs/ \
    --model realesrgan-x4-onnx-cpu \
    --device cpu
```

---

## O que já funciona sem nenhuma instalação adicional

As Fases 1–3 e a Fase 5 (multi-GPU, quando disponível) funcionam com o
ambiente atual. Os ganhos já implementados e ativos:

| Feature | Status | Flag/Configuração |
|---|---|---|
| `torch.compile` (fusão de kernels) | ✅ Ativo (CUDA) | automático em `--device cuda` |
| `cudnn.benchmark` | ✅ Código pronto | ativa com cuDNN instalado (Passo 2) |
| `torch.autocast` FP16 | ✅ Ativo | `--device cuda` + padrão fp32 configurável |
| Tile padding otimizado (32px) | ✅ Ativo | padrão em `tile_pad` |
| Pipeline assíncrono I/O + GPU | ✅ Ativo | `--async-io` |
| Batch inference real (N imagens) | ✅ Ativo | `--batch-size N --async-io` |
| SwinIR runner | ✅ Ativo (`timm` instalado) | `--model swinir-x4` |
| Multi-GPU worker pool | ✅ Implementado | `--multi-gpu` (requer ≥ 2 GPUs) |

---

## Nota sobre Multi-GPU (Fase 5)

O hardware atual tem **1 GPU** (RTX 3070). O flag `--multi-gpu` está
implementado e funcionará automaticamente ao adicionar uma segunda GPU ao
sistema — sem nenhuma alteração de código. Ao detectar menos de 2 GPUs,
o pipeline emite um aviso e continua em modo single-GPU.

---

## Referências

- [NVIDIA cuDNN Installation Guide — Ubuntu](https://docs.nvidia.com/deeplearning/cudnn/latest/installation/linux.html)
- [TensorRT Installation Guide — Ubuntu](https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/installing.html)
- [torch-tensorrt PyPI](https://pypi.org/project/torch-tensorrt/)
- [onnxruntime-gpu PyPI](https://pypi.org/project/onnxruntime-gpu/)
- `docs/plano_otimizacao.md` — plano de otimização completo com fases e ganhos esperados
