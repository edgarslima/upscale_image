# ADR 0014: Backend de Inferência Plugável (TensorRT / ONNX Runtime)

## Status

Aceita

## Contexto

O ADR 0004 estabelece que modelos são adicionados implementando o contrato
`SuperResolutionModel` e registrando-os no registry, sem alterar o pipeline.
Esta mesma extensão é usada para suportar backends de inferência alternativos.

TensorRT (NVIDIA) e ONNX Runtime oferecem 2–4× de throughput adicional sobre
PyTorch compilado ao custo de:
- TensorRT: dependência exclusiva de GPU NVIDIA; engine compilado offline por hardware.
- ONNX Runtime: cross-vendor (AMD, Intel, CPU), mas sem as otimizações específicas
  de hardware que TensorRT oferece em NVIDIA.

Ambos os backends produzem outputs numericamente equivalentes aos do runner PyTorch
(diferenças são sub-pixel, dentro da precisão FP16).

## Decisão

Criar runners separados `TensorRTRunner` e `OnnxRunner` que implementam
`SuperResolutionModel`, carregando engines/modelos compilados exportados offline
por scripts dedicados (`scripts/export_tensorrt.py`, `scripts/export_onnx.py`).

Os runners são registrados no registry com nomes distintos
(`realesrgan-x4-trt-fp16`, `realesrgan-x4-onnx-cuda`) e disponíveis via
`--model` na CLI, sem alterar nenhuma outra parte do sistema.

O registro dos runners TensorRT e ONNX é condicional: ocorre apenas se a dependência
correspondente (`torch-tensorrt`, `onnxruntime-gpu`) estiver instalada. Se não estiver,
o registry simplesmente não expõe esses nomes, e o runner PyTorch padrão continua sendo
o único disponível. Não há erro em tempo de inicialização.

Os scripts de exportação são ferramentas offline que precisam ser executadas uma vez
por modelo e por hardware-alvo. O engine TensorRT exportado é específico para a
arquitetura de GPU onde foi compilado e não pode ser transferido entre hardwares diferentes.

## Consequências

- O pipeline, manifesto, métricas, logs e relatórios não mudam. A abstração do ADR 0004
  é preservada integralmente.
- Runners TensorRT e ONNX são dependências opcionais. Usuários sem GPU NVIDIA podem
  usar ONNX Runtime com CPU EP ou ROCm EP para ganhos menores mas sem dependências pesadas.
- O nome do runner é salvo no manifesto, permitindo identificar qual backend foi usado
  em cada run e comparar resultados entre backends.
- Manter engines TensorRT requer re-exportação ao trocar de GPU ou atualizar o TensorRT.
  Isso é responsabilidade do operador e deve ser documentado no README.
- `torch-tensorrt` e `onnxruntime-gpu` não devem ser instalados no mesmo ambiente
  virtual devido a conflitos de CUDA runtime. Usar `requirements/performance.txt` para
  TensorRT e um ambiente separado para ONNX Runtime se ambos forem necessários.
