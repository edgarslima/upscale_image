# Passo 08: Integrar o Primeiro Runner Real

## Objetivo

Fazer a aplicação executar super-resolution real sobre uma imagem, respeitando device, precisão e pesos.

## Dependências

- Passo 06 concluído
- Passo 07 concluído

## Entregáveis

- Runner real do primeiro modelo
- Carregamento de pesos `.pth`
- Seleção de CPU ou CUDA
- Suporte inicial a `fp32` e `fp16`

## Fluxo técnico mínimo

1. Ler imagem com OpenCV e fallback opcional com Pillow.
2. Converter imagem para tensor.
3. Normalizar input.
4. Executar inferência com `torch.inference_mode()`.
5. Pós-processar output.
6. Converter para imagem e salvar.

## Implementação esperada

1. Carregar pesos a partir da configuração.
2. Resolver device conforme disponibilidade real.
3. Aplicar precisão compatível com o backend.
4. Garantir que o runner funcione isoladamente antes do batch completo.

## Critérios de aceite

- Uma imagem válida gera output ampliado.
- Dimensões de saída são compatíveis com a escala.
- Erros de peso ou device são detectados cedo.

## Como testar

- Uma imagem pequena em CPU.
- O mesmo teste em CUDA, se disponível.
- Peso inexistente.
- Precisão incompatível com o device.

## Armadilhas a evitar

- Integrar o primeiro modelo diretamente dentro do pipeline batch.
- Mascarar erros de device com fallback silencioso não documentado.
- Não liberar memória ou objetos de modelo ao final.
