# Requisitos da POC de Super-Resolution

## Arquivos

- `base.txt`: dependências mínimas da aplicação
- `benchmark.txt`: base + métricas de qualidade
- `dev.txt`: benchmark + testes
- `realesrgan-extra.txt`: extras opcionais do ecossistema Real-ESRGAN

## Conteúdo

### `base.txt`
- `numpy>=2.3,<2.5`
- `torch==2.11.0`
- `torchvision==0.26.0`
- `opencv-python==4.13.0.92`
- `Pillow==12.1.1`
- `typer==0.24.1`
- `rich==14.3.3`
- `PyYAML==6.0.3`

### `benchmark.txt`
- inclui `base.txt`
- `scikit-image==0.26.0`
- `pyiqa==0.1.15.post2`

### `dev.txt`
- inclui `benchmark.txt`
- `pytest==9.0.2`

### `realesrgan-extra.txt`
- inclui `base.txt`
- `basicsr`
- `facexlib`
- `gfpgan`
