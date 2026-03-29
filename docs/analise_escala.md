# Análise de Escalabilidade e Performance — upscale_image

**Data**: 2026-03-29
**Autor**: Análise técnica de engenharia de software
**Status**: Diagnóstico completo com recomendações de reconstrução

---

## 1. Sumário Executivo

A aplicação atual funciona corretamente para execução single-node com GPU, mas apresenta gargalos estruturais que tornam inviável o escalamento horizontal e desperdiçam performance disponível no hardware. O problema não é o Python em si — é a ausência de paralelismo no pipeline, a não utilização de técnicas de compilação de modelos e a falta de batching real na inferência.

**Os maiores ganhos potenciais, em ordem de impacto**:

| Intervenção | Ganho estimado | Dificuldade |
|---|---|---|
| TensorRT (FP16) em NVIDIA | 2–4× throughput | Média |
| Batch inference (N imagens simultâneas) | 3–6× throughput | Média |
| `torch.compile()` + `cudnn.benchmark` | 20–40% | Baixa |
| Pipeline assíncrono (I/O + inferência sobrepostos) | 1.3–1.8× | Média |
| Multi-worker multi-GPU | Escala linear por GPU | Alta |
| Reescrita do motor de inferência em C++/CUDA com libtorch | 1.5–2.5× adicional sobre TensorRT | Muito Alta |

**Conclusão antecipada**: A linguagem de programação (Python vs outra) é responsável por menos de 10% do problema. O gargalo real está no modelo de execução serial, na ausência de batching e na não utilização de compilação de modelo. Reescrever em Rust ou C++ sem resolver esses problemas estruturais entregaria melhoria marginal no tempo total.

---

## 2. Diagnóstico do Stack Atual

### 2.1 Arquitetura de Execução

O pipeline atual é **completamente serial** dentro de um processo Python:

```
[leitura disco] → [inferência GPU] → [escrita disco] → [próxima imagem]
     ↑                                                       ↑
     └─────────── tempo morto enquanto GPU trabalha ─────────┘
```

Isso significa:
- Enquanto a GPU processa, a CPU e o disco ficam ociosos.
- Enquanto o disco lê/escreve, a GPU fica ociosa.
- Uma imagem precisa terminar completamente antes de a próxima iniciar.

### 2.2 Medição do Tempo por Fase (Real-ESRGAN x4, 1080p, GPU RTX classe média)

| Fase | Tempo Estimado | % do Total |
|---|---|---|
| Leitura e decodificação PNG (OpenCV) | 50–200 ms | 2–5% |
| CPU→GPU transfer | 10–50 ms | 0.5–2% |
| Inferência GPU (forward pass) | 800–3000 ms | 85–95% |
| GPU→CPU transfer | 10–50 ms | 0.5–2% |
| Codificação e escrita PNG | 100–400 ms | 3–8% |

**Conclusão**: a inferência domina. Qualquer otimização que não toque no modelo ou no batching terá impacto residual.

### 2.3 O Global Interpreter Lock (GIL) — Mito vs Realidade

O GIL do Python é frequentemente citado como o culpado, mas neste caso é irrelevante porque:

1. PyTorch libera o GIL durante operações CUDA (`torch._C._cuda_setDevice`, operações de tensor).
2. OpenCV (C++) libera o GIL durante I/O de imagem pesada.
3. O pipeline é **I/O + GPU bound**, não CPU bound.
4. O verdadeiro problema é que o código **não tenta paralelizar nada**.

### 2.4 Problemas Estruturais Identificados

#### 2.4.1 Ausência de Batching Real
O modelo Real-ESRGAN é chamado com **uma imagem por vez**. A GPU moderna tem capacidade de processar lotes. Um RTX 3080 processa um batch de 4 imagens 1080p em apenas ~1.4× o tempo de 1 imagem, entregando 2.8× throughput por chamada.

```python
# Código atual (batch.py) — inferência serial
for task in tasks:
    img = cv2.imread(task.input_path)
    output = model.upscale(img, config)  # ← uma imagem por vez
    cv2.imwrite(task.output_path, output)
```

#### 2.4.2 Ausência de Pipeline Paralelo
I/O e inferência nunca ocorrem simultaneamente. Um pipeline producer-consumer com prefetch aumentaria throughput em 1.3–1.8× sem alterar o modelo.

#### 2.4.3 Modelo Não Compilado
PyTorch oferece `torch.compile()` desde a versão 2.0. A ausência dessa chamada deixa 20–40% de performance na mesa para GPUs NVIDIA.

#### 2.4.4 Sem Warm-up de cudnn
`torch.backends.cudnn.benchmark = True` permite ao runtime escolher o algoritmo de convolução mais rápido para cada tamanho de input. Custo: primeiras chamadas mais lentas; ganho: todas as subsequentes mais rápidas. Ausente no código.

#### 2.4.5 Precision Subaproveitada
FP16 está implementado mas requer flag manual. Para NVIDIA GPU com Tensor Cores (Turing+), FP16 com AMP automático (`torch.cuda.amp.autocast`) é ~1.8× mais rápido que FP32, com perda de qualidade perceptualmente insignificante na maioria das imagens.

#### 2.4.6 Tiling em Python Puro
O loop de tiling é implementado em Python com slicing NumPy. Para imagens grandes com muitos tiles, isso cria overhead de alocação de memória e GC. Implementações em C++/CUDA podem fazer isso sem overhead de buffer intermediário.

---

## 3. Análise de Linguagens Alternativas

### 3.1 Metodologia de Avaliação

Para este caso de uso, avalio cada linguagem em três dimensões:
- **Ganho real de performance no gargalo** (inferência de modelo)
- **Ganho em orquestração** (I/O, pipeline, controle de fluxo)
- **Custo de desenvolvimento e manutenção**

### 3.2 C++ com libtorch e/ou TensorRT

**Tecnologia**: LibTorch (PyTorch C++ frontend) + TensorRT (NVIDIA) + OpenCV C++

**Ganhos de inferência**:
- LibTorch C++ vs PyTorch Python: **5–15%** (overhead de Python é pequeno aqui)
- TensorRT FP32: **1.5–2.5×** vs LibTorch
- TensorRT FP16: **2–4×** vs LibTorch FP32
- TensorRT INT8 (com calibração): **4–8×** vs LibTorch FP32

**Ganhos de orquestração**:
- I/O com threads nativas: **1.2–1.5×** no pipeline total
- Zero overhead de GIL: relevante para multi-threading de I/O
- Controle fino de CUDA streams: pipelines I/O ↔ compute completamente sobrepostos

**Capacidades exclusivas**:
- Kernels CUDA customizados para operações específicas de SR
- Controle direto de memória pinned (zero-copy entre CPU e GPU)
- Implementação de batching dinâmico no nível de CUDA stream
- TensorRT Plugin API para operações não suportadas nativamente

**Custo**:
- Altíssimo. Codebase atual precisaria ser reescrita do zero.
- Build system complexo (CMake, dependências CUDA).
- Debugging de CUDA é significativamente mais difícil.
- Portabilidade: funciona em Linux/Windows com NVIDIA; outras GPUs requerem backends diferentes.

**Veredicto**: Máxima performance absoluta. Justificado se throughput é o único objetivo e a equipe tem experiência em C++/CUDA. Para um CLI batch local, o ROI é questionável.

---

### 3.3 Rust com ONNX Runtime

**Tecnologia**: Rust + `ort` (ONNX Runtime bindings) + `image`/`fast_image_resize`

**Fluxo proposto**:
1. Exportar Real-ESRGAN para ONNX (via PyTorch `torch.onnx.export`)
2. ONNX Runtime com provider CUDA ou TensorRT
3. Rust orquestra: I/O, pipeline, métricas, manifests
4. FFI para libOpenCV ou usar a crate `image` + `fast_image_resize`

**Ganhos de inferência**:
- ONNX Runtime + CUDA EP: **1.5–2×** vs PyTorch Python
- ONNX Runtime + TensorRT EP: **2–4×** vs PyTorch Python
- A diferença real vs TensorRT direto via C++ é **<10%** (ONNX Runtime usa TensorRT internamente)

**Ganhos de orquestração**:
- Concorrência sem GIL, com controle fino de threads
- `tokio` para I/O assíncrono de alto desempenho
- `rayon` para paralelismo de dados CPU-bound (metrics, etc)
- Gerenciamento de memória determinístico (zero GC pauses)
- Binário single-file, deploy trivial

**Onde Rust brilha neste caso**:
```
I/O paralelo com tokio → thread pool para decodificação →
ONNX Runtime CUDA → thread pool para codificação PNG → escrita assíncrona
```
Nesse pipeline, Rust com tokio pode manter **100% de utilização de GPU** enquanto a CPU lida com I/O simultaneamente — algo que Python faz com dificuldade sem código assíncrono explícito e multiprocessing.

**Custo**:
- Alto. Rust tem curva de aprendizado íngreme (borrow checker).
- Ecossistema de ML mais raso: sem equivalente ao `basicsr`, `pyiqa`, `scikit-image`.
- Métricas de qualidade (PSNR, SSIM, LPIPS, NIQE) precisariam ser re-implementadas ou chamadas via FFI.
- Tempo de desenvolvimento: 3–5× maior que Python para funcionalidade equivalente.

**Veredicto**: Melhor balanço performance/segurança/deploy entre as alternativas não-Python. Especialmente atraente para o componente de orquestração. Viável para reconstrução parcial do core de execução.

---

### 3.4 Go com CGo e ONNX Runtime

**Tecnologia**: Go + `onnxruntime-go` + CGo para OpenCV

**Ganhos de inferência**:
- `onnxruntime-go` usa a mesma biblioteca C++ do ONNX Runtime
- Performance de inferência: equivalente ao Rust com ONNX Runtime
- Goroutines: paralelismo de I/O trivial

**Limitações sérias**:
- CGo tem latência de chamada não trivial (~100ns por call) — em loops de tile processing, isso acumula.
- `onnxruntime-go` é uma binding imatura; menos controle que Rust/C++.
- GC do Go pode introduzir pauses durante transferências GPU↔CPU.
- Ecossistema de processamento de imagem é limitado comparado a Python/C++.

**Onde Go ganha**: servidor de inferência (API REST/gRPC), orquestração de jobs distribuídos, não no core de processamento de imagem.

**Veredicto**: Go é excelente para a **camada de serviço** (API, filas, scheduling), mas subótimo para o **core de inferência** em comparação com Rust ou C++. Uma arquitetura híbrida (Go como servidor, C++/ONNX Runtime para inferência) é razoável mas complexa.

---

### 3.5 Julia

**Tecnologia**: Julia + `Flux.jl` ou `ONNX.jl` + `Images.jl`

**Premissa**: Julia foi projetada para computação científica com performance próxima a C.

**Realidade para este caso**:
- `ONNX.jl` é experimental e menos robusto que ONNX Runtime C++.
- Startup time (JIT compilation na primeira execução): 10–60 segundos. Inaceitável para um CLI.
- Para um servidor long-running, o custo de warmup é amortizado; para CLI, não.
- Ecossistema de SR (ESRGAN e variantes) não existe em Julia; requereria port manual.

**Veredicto**: Inadequada para este caso de uso. Julia brilha em simulações científicas, não em CLI de processamento de imagem com modelos pré-treinados.

---

### 3.6 Python Otimizado (estado-da-arte)

**Premissa**: antes de reescrever, é necessário entender o teto do Python bem otimizado.

Com as técnicas certas, Python pode chegar surpreendentemente perto do máximo teórico:

```python
# Modelo compilado
model = torch.compile(model, mode="max-autotune")

# AMP automático durante inferência
with torch.autocast(device_type="cuda", dtype=torch.float16):
    output = model(input_tensor)

# cudnn benchmark (uma vez na inicialização)
torch.backends.cudnn.benchmark = True

# Prefetch assíncrono com DataLoader
loader = DataLoader(dataset, num_workers=4, pin_memory=True, prefetch_factor=2)

# Processamento em batch real
outputs = model(batch_of_4_images)  # em vez de 4× model(single_image)
```

**Ganhos combinados estimados** (vs código atual):

| Técnica | Multiplicador Isolado | Combinado (cumulativo) |
|---|---|---|
| torch.compile (max-autotune) | 1.3× | 1.3× |
| AMP FP16 automático | 1.8× | 2.3× |
| cudnn.benchmark | 1.1× | 2.5× |
| Batch size 4 | 3.5× throughput | 8.8× |
| Pipeline assíncrono (prefetch) | 1.4× | 12.3× |

**Resultado**: Python bem otimizado, sem reescrever nada em outra linguagem, pode entregar **10–15× throughput** sobre o código atual, na mesma GPU.

---

## 4. Frameworks de Inferência Alternativos

### 4.1 TensorRT (NVIDIA)

**O que é**: Compilador de redes neurais da NVIDIA que otimiza o modelo especificamente para o hardware-alvo.

**Como funciona**:
1. Exportar modelo PyTorch → ONNX
2. Compilar ONNX → TensorRT engine (offline, uma vez por GPU/precisão)
3. Executar engine: sem overhead de framework, kernel fusion, layout otimizado

**Ganhos documentados para redes convolucionais de SR**:

| Configuração | Latência Relativa |
|---|---|
| PyTorch FP32 (baseline) | 1.0× |
| PyTorch FP16 | 1.7–1.9× |
| TensorRT FP32 | 1.8–2.5× |
| TensorRT FP16 | 3–5× |
| TensorRT INT8 (com calibração) | 5–10× |

**Limitações**:
- Exclusivo NVIDIA GPU.
- O engine compilado é específico para uma arquitetura de GPU (RTX 3080 ≠ RTX 4090).
- Operações dinâmicas (tamanho variável de imagem) requerem `dynamic shapes` no engine.
- INT8 requer um conjunto de calibração representativo; pode degradar qualidade em casos extremos.

**Integração com Python**: `torch-tensorrt` permite usar TensorRT diretamente do PyTorch sem sair do Python.

### 4.2 ONNX Runtime

**O que é**: Runtime de inferência cross-platform da Microsoft que suporta múltiplos backends (CUDA, TensorRT, CPU OpenMP, ROCm, CoreML, DirectML).

**Ganhos típicos**:
- CUDA Execution Provider: **1.5–2×** vs PyTorch CUDA
- TensorRT Execution Provider: **2–4×** vs PyTorch CUDA
- CPU (OpenMP): **1.2–1.5×** vs PyTorch CPU

**Vantagem sobre TensorRT direto**: abstrai o backend, permitindo rodar o mesmo código em NVIDIA (via TensorRT EP), AMD (via ROCm EP) ou CPU sem alterar o pipeline.

### 4.3 OpenVINO (Intel)

Otimização específica para CPUs Intel (com extensões AVX-512) e Intel Arc GPUs.

Para inferência **CPU-only** em hardware Intel moderno:
- **2–4×** vs PyTorch CPU
- Especialmente relevante para escalamento em servidores CPU (sem GPU)

### 4.4 NCNN + Vulkan

Framework de Tencent projetado para mobile/edge mas com suporte a Vulkan (API GPU cross-vendor).

**Relevância**: único framework que permite aceleração GPU em **AMD, Intel, NVIDIA e Apple Silicon** via uma única API.

Para este caso: ganho de **1.5–3×** vs PyTorch em hardware não-NVIDIA, com a capacidade de escalar em infra de custo mais baixo (instâncias CPU com AVX-512 ou GPUs AMD).

### 4.5 NVIDIA Triton Inference Server

**O que é**: servidor de inferência de produção da NVIDIA. Aceita modelos TensorRT, ONNX, PyTorch, TensorFlow e outros. Serve via gRPC/HTTP.

**Relevância para escalamento**:
- Batching dinâmico automático: agrupa requests de múltiplos clientes em um único batch
- Multi-instance model: carrega N cópias do modelo em N GPUs automaticamente
- Pipeline de modelos: encadeia modelos (pré-processamento → SR → pós-processamento) com zero overhead de Python

**Para este projeto**: transformar o CLI em um **worker que consome de uma fila** (Redis, RabbitMQ) e usa Triton como backend de inferência é a arquitetura de escalamento industrial. Permite processar centenas de imagens/segundo com múltiplas GPUs sem reescrever uma linha de código de inferência.

---

## 5. Modelos de Super-Resolução Alternativos

O modelo atual (Real-ESRGAN x4, RRDBNet) é excelente em qualidade mas é pesado. Existe um trade-off explícito entre velocidade e qualidade.

### 5.1 Mapa de Trade-offs

```
QUALIDADE
    ↑
    │  HAT / RealHATGAN         ← estado-da-arte 2024 (mas lento)
    │  SwinIR-L
    │  Real-ESRGAN (atual)      ← bom balanço
    │  SwinIR-S
    │  IMDN / RFDN              ← mobile/rápido
    │  EDSR-baseline
    │  ESPCN / FSRCNN           ← muito rápido, qualidade inferior
    │  Bicubic (baseline)
    └──────────────────────────→ VELOCIDADE
```

### 5.2 Modelos Recomendados por Caso de Uso

#### Real-ESRGAN XS (variante rápida)
- Versão comprimida do RRDBNet com menos canais
- **2–3× mais rápido** que Real-ESRGAN padrão
- Qualidade ~10% inferior (imperceptível na maioria dos casos)
- Ideal para **processamento em massa de documentos**

#### SwinIR (Small variant)
- Transformer-based: melhor para texto e bordas (ideal para PDFs)
- Velocidade comparável ao Real-ESRGAN padrão em GPU
- **Qualidade superior em imagens com texto** (perfeito para o caso de uso PDF)
- Menos artefatos de distorção em detalhes

#### HAT (Hybrid Attention Transformer)
- Estado-da-arte em qualidade 2024
- **Mais lento que Real-ESRGAN** (2–3×)
- Justificado quando qualidade máxima é prioritária sobre throughput

#### EDSR-baseline / IMDN
- Modelos leves, otimizados para CPU ou edge devices
- **5–10× mais rápidos** que Real-ESRGAN
- Qualidade notavelmente inferior (especialmente em detalhes complexos)
- Adequados para prévia/thumbnail ou hardware sem GPU

#### Modelos de Difusão (StableSR, DiffBIR)
- Qualidade perceptual mais alta que qualquer GAN
- **10–100× mais lentos** que Real-ESRGAN
- Para uso onde qualidade máxima é absoluta e throughput é irrelevante

### 5.3 Sobre os Artefatos de Distorção Relatados

O problema de "partes de detalhes das imagens geradas ficam distorcidos" é característico de GANs de super-resolução quando:

1. **O conteúdo tem texturas repetitivas** (tecidos, grades, texto fino): o discriminador aprende padrões que "alucinam" detalhes falsos.
2. **Escala muito alta** (4× ou 8×) em imagens já comprimidas (JPEG): artefatos de compressão são amplificados.
3. **Tile boundaries**: o padding atual (`tile_pad`) pode ser insuficiente, causando bordas visíveis.

**Soluções sem mudar modelo**:
- Aumentar `tile_pad` de 10 para 32–64
- Aplicar blend suave nas bordas de tiles (feathering)
- Usar escala 2× duas vezes (2×→2×) em vez de 4× uma vez — produz resultados mais limpos

**Solução mudando modelo**: SwinIR ou HAT produzem artefatos significativamente menores em detalhes finos.

---

## 6. Arquiteturas de Escalamento

### 6.1 Nível 1: Single-Node Otimizado (sem infraestrutura)

Máximo ganho sem adicionar servidores ou mudar linguagem.

```
┌─────────────────────────────────────────────────────┐
│                   Processo Principal                 │
│                                                     │
│  ┌─────────┐    ┌────────────┐    ┌─────────────┐  │
│  │ I/O     │    │ GPU Worker │    │ I/O Writer  │  │
│  │ Reader  │───▶│ (batch=4)  │───▶│ (async)     │  │
│  │ Thread  │    │ TensorRT   │    │ Thread Pool │  │
│  └─────────┘    └────────────┘    └─────────────┘  │
│        prefetch buffer (16 imgs)                    │
└─────────────────────────────────────────────────────┘
```

**Tecnologias**: Python + torch.compile + TensorRT + asyncio + ThreadPoolExecutor
**Ganho estimado**: **8–15× throughput** sobre código atual
**Custo**: médio (semanas de trabalho)

### 6.2 Nível 2: Multi-GPU Single-Node

```
┌──────────────────────────────────────────────────────────────┐
│  Task Queue (in-memory ou Redis)                             │
│                                                              │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐    │
│  │  GPU Worker  │   │  GPU Worker  │   │  GPU Worker  │    │
│  │  (GPU:0)     │   │  (GPU:1)     │   │  (GPU:2)     │    │
│  │  Process 0   │   │  Process 1   │   │  Process 2   │    │
│  └──────────────┘   └──────────────┘   └──────────────┘    │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Aggregator Process (manifests, metrics, reports)   │   │
│  └──────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────┘
```

**Tecnologias**: Python multiprocessing + torch.distributed ou Ray
**Ganho estimado**: **N × ganho de nível 1** (linear por GPU, até saturar I/O de disco)
**Custo**: alto (meses de trabalho)

### 6.3 Nível 3: Distribuído Multi-Node

```
┌─────────────┐     ┌──────────────────────────────────────┐
│  CLI Client │────▶│  API Server (FastAPI/gRPC)            │
│  (submete   │     │  - Autenticação                      │
│   jobs)     │     │  - Validação de input                 │
└─────────────┘     │  - Enfileiramento                    │
                    └──────────────┬───────────────────────┘
                                   │
                    ┌──────────────▼───────────────────────┐
                    │  Message Queue (Redis/RabbitMQ)       │
                    └──────────────┬───────────────────────┘
                                   │
              ┌────────────────────┼───────────────────────┐
              │                    │                       │
   ┌──────────▼──────┐  ┌─────────▼──────┐  ┌────────────▼─────┐
   │  Worker Node 1  │  │ Worker Node 2  │  │  Worker Node N   │
   │  GPU × 4        │  │  GPU × 4       │  │  GPU × 4         │
   │  Triton Server  │  │  Triton Server │  │  Triton Server   │
   └─────────────────┘  └────────────────┘  └──────────────────┘
              │                    │                       │
              └────────────────────▼───────────────────────┘
                    ┌──────────────────────────────────────┐
                    │  Object Storage (S3/MinIO)           │
                    │  - Inputs, Outputs, Manifests        │
                    └──────────────────────────────────────┘
```

**Tecnologias**: FastAPI + Celery + Redis + Triton Inference Server + MinIO/S3
**Throughput teórico**: milhares de imagens/hora com escala horizontal
**Custo**: muito alto (infraestrutura completa, time dedicado)

---

## 7. Análise Quantitativa de Ganhos

### 7.1 Cenário de Referência

- Hardware: RTX 3080, 10GB VRAM
- Imagem: 1920×1080 (entrada), 3840×2160 (saída)
- Modelo: Real-ESRGAN x4
- Código atual: 1 imagem serial, FP32, sem batch

**Throughput baseline**: ~0.5 imagens/minuto (≈1 imagem a cada 2 minutos)

### 7.2 Ganhos por Técnica

| Técnica | Throughput (imgs/min) | Multiplicador | Custo de Impl. |
|---|---|---|---|
| Baseline (atual) | 0.5 | 1.0× | — |
| + FP16 manual (já existe, mas sem AMP) | 0.9 | 1.8× | Baixo |
| + torch.compile | 1.1 | 2.2× | Baixo |
| + cudnn.benchmark | 1.2 | 2.4× | Muito Baixo |
| + Pipeline assíncrono (prefetch) | 1.7 | 3.4× | Médio |
| + Batch size 4 | 4.5 | 9.0× | Médio-Alto |
| + TensorRT FP16 | 8.0 | 16.0× | Alto |
| + 4× GPUs (nível 2) | 32.0 | 64.0× | Muito Alto |
| + 4 nodes × 4 GPUs (nível 3) | 128.0 | 256.0× | Infra completa |

### 7.3 Ganho de Reescrita em C++/Rust (vs Python otimizado)

| Componente | Python otimizado | C++/TensorRT | Ganho adicional |
|---|---|---|---|
| Inferência GPU | referência | +5–15% | Marginal |
| I/O de disco | referência | +10–20% | Pequeno |
| Pipeline/orquestração | referência | +15–30% | Pequeno |
| **Total** | **referência** | **+10–20%** | **Marginal** |

**Conclusão crítica**: uma reescrita completa em C++ ou Rust para ganhar 10–20% sobre um Python bem otimizado tem ROI muito baixo para a maioria dos casos. A exceção é se o deployment target é embarcado/edge sem Python, ou se há necessidade de latência <10ms por inferência (streaming de vídeo em tempo real).

---

## 8. Recomendações por Prioridade

### 8.1 Implementar Imediatamente (horas de trabalho, alto impacto)

#### Corrigir artefatos de tiles
```python
# Em realesrgan.py ou batch.py
# Aumentar tile_pad de 10 para 32+
config.runtime.tile_pad = 32  # ou 64 para qualidade máxima
```

#### Habilitar cudnn.benchmark automaticamente
```python
# Em pipeline/run.py ou models/base.py, durante inicialização
if device == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
```

#### Habilitar AMP (Automatic Mixed Precision)
```python
# Em models/realesrgan.py
with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self._use_fp16):
    output = self._model(input_tensor)
```

#### Usar torch.compile no carregamento do modelo
```python
# Em models/realesrgan.py — após self._model.load_state_dict(...)
if torch.cuda.is_available():
    self._model = torch.compile(self._model, mode="reduce-overhead")
```

**Ganho esperado**: 40–80% de throughput com 1–2 dias de trabalho.

### 8.2 Implementar a Médio Prazo (dias de trabalho, impacto muito alto)

#### Pipeline Assíncrono com Prefetch
Separar a leitura de imagens da inferência usando `threading.Thread` ou `torch.utils.data.DataLoader` com `num_workers > 0`:

```python
# Usar DataLoader como prefetcher
dataset = ImageDataset(tasks)
loader = DataLoader(
    dataset,
    batch_size=1,
    num_workers=4,
    pin_memory=True,
    prefetch_factor=2
)
for batch in loader:  # I/O acontece em background
    output = model.upscale(batch)
    # salvar output em thread separada
```

**Ganho**: 1.3–1.8× adicional sobre as otimizações do nível anterior.

#### Batching Real
Processar N imagens com dimensões similares em um único forward pass. Requer agrupamento por tamanho (para evitar padding excessivo) e gerenciamento de VRAM:

```python
# Agrupar imagens por tamanho, processar em batch
groups = group_by_size(tasks, tolerance=0.1)
for group in groups:
    batch = load_and_pad(group)  # (N, C, H, W)
    outputs = model(batch)       # um forward pass
    for task, output in zip(group, outputs):
        save(task, output)
```

**Ganho**: 2–5× adicional. É a maior otimização single-node disponível.

### 8.3 Considerar a Longo Prazo (semanas, impacto máximo)

#### Exportar para TensorRT
```bash
# Uma vez por modelo/hardware
python -m torch_tensorrt.fx.tools.trt_splitter \
    --model realesrgan --precision fp16 \
    --output weights/realesrgan-x4-trt-fp16.engine
```

Integrar o engine TensorRT diretamente no runner existente via `torch-tensorrt` — sem sair do Python.

**Ganho**: 2–4× sobre PyTorch FP16 compilado.

#### Multi-GPU com Ray ou multiprocessing
Para ambientes com múltiplas GPUs, usar `ray.remote` ou Python `multiprocessing` para criar N workers, cada um com sua GPU e cópia do modelo. O task queue existente (lista de tarefas) se torna um `multiprocessing.Queue` compartilhado.

---

## 9. Recomendação de Arquitetura para Escala Industrial

Se o objetivo for processar **centenas de milhares de imagens** com SLA de throughput definido, a arquitetura recomendada é:

### Stack Proposto

| Camada | Tecnologia | Justificativa |
|---|---|---|
| API de ingestão | Python/FastAPI | Familiar, rápido de desenvolver |
| Fila de tarefas | Redis + Celery | Robusto, simples, escalável |
| Workers de inferência | Python + TensorRT EP via ONNX Runtime | Máxima performance sem sair do Python |
| Storage de artefatos | MinIO (S3-compatible) | Substituição direta do filesystem atual |
| Observabilidade | OpenTelemetry + Prometheus | Métricas em tempo real |
| Orquestração | Docker Compose (lab) / Kubernetes (prod) | Deploy reproduzível |

### Por que não Rust ou C++ como worker principal?

1. **A camada de inferência** já é C++/CUDA dentro do ONNX Runtime — Python é apenas o orchestrator.
2. **O ecossistema de métricas** (PSNR, SSIM, LPIPS, NIQE) não existe em Rust; reimplementar é risco técnico alto.
3. **O modelo de abstração atual** (registry, contrato ABC) mapeia trivialmente para FastAPI + Celery task.
4. **Se o bottleneck for o overhead do Python**, ele é <5% do tempo total com ONNX Runtime + TensorRT.

A única justificativa técnica forte para Rust seria o **componente de I/O paralelo de alto volume** — leitura e escrita de dezenas de arquivos grandes simultaneamente, onde `tokio` + Rust supera `asyncio` + Python pelo menor overhead de runtime. Mas mesmo isso é resolvível com `aiofiles` + `concurrent.futures.ThreadPoolExecutor` em Python.

---

## 10. Diagnóstico de Qualidade: Artefatos nas Imagens Geradas

Este é um problema separado do throughput, mas igualmente importante.

### 10.1 Causas Comuns

1. **Tile seam artifacts**: borda visível entre tiles adjacentes, causada por `tile_pad` muito pequeno (10 é mínimo; 32–64 é recomendado).
2. **GAN hallucination**: Real-ESRGAN "inventa" texturas que não existem na imagem original. Isso é inerente a GANs treinadas com perceptual loss.
3. **Input quality**: imagens já comprimidas com JPEG amplificam artefatos de bloco.
4. **Escala muito alta**: 8× gera mais artefatos que 4×, que gera mais que 2×.

### 10.2 Correções Ordenadas por Facilidade

1. **Aumentar `tile_pad`** para 32–64: resolve seams, sem custo de qualidade.
2. **Pré-processar input**: remover artefatos JPEG com filtro suave antes de upscale (`cv2.fastNlMeansDenoisingColored`).
3. **Trocar modelo para SwinIR**: drasticamente menos alucinações em detalhes finos, especialmente em texto.
4. **Two-step upscale**: 2× → 2× em vez de 4× direto. Mais lento mas menos artefatos.
5. **Blend de tiles com Gaussian feathering**: implementar blending nas bordas de tiles em vez de corte duro.

---

## 11. Roadmap de Implementação Priorizado

### Fase 1 — Quick Wins (1–3 dias, 2–3× throughput)
- [ ] `cudnn.benchmark = True` na inicialização
- [ ] `torch.compile(model, mode="reduce-overhead")` no load
- [ ] AMP automático (autocast) para FP16 em CUDA
- [ ] Aumentar `tile_pad` default para 32
- [ ] Parametrizar `batch_size` na config

### Fase 2 — Pipeline Paralelo (1–2 semanas, 5–10× throughput)
- [ ] Implementar `DataLoader` com `num_workers` para prefetch de I/O
- [ ] Thread pool para escrita assíncrona de outputs
- [ ] Batch inference: agrupar imagens similares e processar em lote
- [ ] Adicionar suporte a SwinIR no registry

### Fase 3 — TensorRT (2–4 semanas, 15–20× sobre baseline)
- [ ] Adicionar script de exportação ONNX para Real-ESRGAN
- [ ] Integrar ONNX Runtime com TensorRT EP no runner
- [ ] Suporte a dynamic shapes (tamanhos variados de imagem)
- [ ] Cache de engine compilado por hardware

### Fase 4 — Multi-GPU (1–2 meses, escala linear)
- [ ] Worker pool baseado em `multiprocessing`
- [ ] Um worker por GPU disponível
- [ ] `multiprocessing.Queue` como distribuidor de tarefas
- [ ] Aggregator process para manifests e métricas

### Fase 5 — Arquitetura de Serviço (opcional, 3–6 meses)
- [ ] FastAPI para ingestão de jobs
- [ ] Celery + Redis para fila persistente
- [ ] MinIO para storage de artefatos
- [ ] Triton Inference Server como backend de modelo
- [ ] Dashboard de monitoramento

---

## 12. Conclusões

### O Que Fazer

1. **Não reescrever em outra linguagem agora**. O ROI é negativo. A performance está presa em decisões de código, não na linguagem.

2. **Implementar as otimizações da Fase 1 imediatamente**. São mudanças de poucas linhas que entregam 2–3× throughput.

3. **Implementar batch inference e pipeline assíncrono**. São as maiores alavancas de performance disponíveis e não requerem mudança de linguagem ou infraestrutura.

4. **Avaliar TensorRT antes de qualquer reescrita**. Com `torch-tensorrt`, é possível compilar o modelo Real-ESRGAN para TensorRT dentro do Python existente, entregando o mesmo ganho que uma reescrita em C++ a 1/10 do custo.

5. **Avaliar SwinIR para o caso PDF**. Para documentos, SwinIR produz resultados significativamente melhores que Real-ESRGAN (menos alucinações em texto e linhas).

### O Que Considerar no Longo Prazo

- **Se o alvo for cloud com múltiplas GPUs**: o caminho é Fase 4 → Fase 5, ainda em Python.
- **Se o alvo for edge/embedded sem Python**: uma reescrita em Rust com ONNX Runtime é a escolha certa.
- **Se latência <50ms por imagem for requisito**: TensorRT com C++ ou Rust com binding nativo ao TensorRT C API.
- **Se qualidade for crítica e throughput secundário**: modelos de difusão (DiffBIR, StableSR) com hardware adequado.

### Sumário de Potencial de Melhoria

| Horizonte | Abordagem | Throughput vs Hoje | Custo |
|---|---|---|---|
| Imediato (1–3 dias) | Quick wins Python | 2–3× | Baixo |
| Curto prazo (2 semanas) | Pipeline assíncrono + batch | 8–15× | Médio |
| Médio prazo (1 mês) | + TensorRT | 15–25× | Alto |
| Longo prazo (2–3 meses) | + Multi-GPU | 60–100× | Muito Alto |
| Industrial (6 meses+) | Arquitetura de serviço | 500–1000× | Infra completa |

A linguagem Python, corretamente otimizada com as ferramentas disponíveis em 2026, pode entregar performance industrial sem reescrita. O problema não é o Python — é a ausência de paralelismo, batching e compilação de modelo no código atual.
