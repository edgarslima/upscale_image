# ADR 0007: Stack Tecnológica Principal

## Status

Aceita

## Contexto

O sistema exige inferência com deep learning, manipulação de imagem, CLI estruturada e cálculo de métricas técnicas.

## Decisão

Adotar a seguinte stack principal:

- Python 3.12+
- PyTorch para inferência, device e precisão
- OpenCV como biblioteca principal de IO de imagem
- Pillow como fallback e suporte complementar
- Typer para CLI
- Rich para logs e saída estruturada
- PyYAML para configuração
- NumPy para base numérica
- scikit-image para PSNR e SSIM
- pyiqa para LPIPS e NIQE

Adições opcionais para otimização de performance (não obrigatórias para o funcionamento base):

- `torch.compile` e `torch.backends.cudnn.benchmark`: aceleração de inferência em CUDA,
  ativados automaticamente quando `device=cuda` (ADR 0011). Fazem parte do PyTorch
  instalado — sem dependência adicional.
- `torch.autocast`: substitui a conversão manual de precisão (`.half()`), gerenciando
  automaticamente quais operações usam FP16 (ADR 0011). Disponível no PyTorch instalado.
- `torch-tensorrt` (opcional): compilação TensorRT para máxima performance em GPU NVIDIA.
  Instalado via `requirements/performance.txt`. Expõe runners adicionais no registry
  (ADR 0014).
- `onnxruntime-gpu` (opcional): backend de inferência cross-vendor. Alternativa ao
  torch-tensorrt para ambientes sem GPU NVIDIA exclusiva (ADR 0014).
- `timm` (opcional): suporte ao modelo SwinIR. Instalado via `requirements/swinir.txt`.

## Consequências

- A implementação deve respeitar responsabilidades de cada biblioteca.
- OpenCV permanece o caminho principal de leitura e escrita.
- PyTorch centraliza controle de CPU/CUDA. `torch.autocast` substitui `.half()` manual.
- Dependências opcionais são carregadas condicionalmente; sua ausência nunca causa erro
  em tempo de inicialização — apenas reduz os runners disponíveis no registry.
- `torch-tensorrt` e `onnxruntime-gpu` não devem coexistir no mesmo ambiente virtual
  devido a conflitos de CUDA runtime.
