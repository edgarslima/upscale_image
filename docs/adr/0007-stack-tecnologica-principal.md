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

## Consequências

- A implementação deve respeitar responsabilidades de cada biblioteca.
- OpenCV permanece o caminho principal de leitura e escrita.
- PyTorch centraliza controle de CPU/CUDA e `fp32/fp16`.
