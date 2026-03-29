# ADR 0011: Compilação de Modelo e AMP Automático

## Status

Aceita

## Contexto

O pipeline atual carrega o modelo em modo eager (PyTorch padrão) e aplica precisão mista
de forma manual via `.half()` nos tensores. Isso deixa dois ganhos de performance intocados:

1. `torch.compile()` aplica fusão de kernels e otimizações de grafo de computação,
   entregando 20–40% de redução de latência em GPUs NVIDIA sem alterar o comportamento
   observável do modelo.

2. `torch.autocast` gerencia automaticamente quais operações são executadas em FP16
   (convoluções, matmuls) e quais permanecem em FP32 (operações numericamente sensíveis),
   eliminando o risco de `nan` silencioso causado pela conversão manual.

3. `torch.backends.cudnn.benchmark = True` habilita o autotuner do cuDNN, que seleciona
   o algoritmo de convolução mais rápido para cada tamanho de tensor encontrado na primeira
   execução.

## Decisão

Aplicar `torch.compile()`, `torch.autocast` e `torch.backends.cudnn.benchmark = True`
em todos os runners que executam em CUDA, como comportamento padrão ativado automaticamente
quando `device=cuda`.

A ativação é feita no método `load()` do runner, após `net.eval()`, condicionada a
`torch.cuda.is_available()`. Runners em CPU não são afetados.

A conversão manual via `.half()` nos tensores é substituída integralmente por `torch.autocast`.
O modelo permanece em FP32 em memória; `autocast` gerencia a precisão por operação.

## Consequências

- Runners CUDA produzem resultados numericamente equivalentes aos produzidos antes da
  mudança, com menor latência de inferência.
- `torch.compile` transforma `self._net` em `torch._dynamo.OptimizedModule`.
  Verificações de tipo interno nos testes que usam `isinstance(._net, ConcreteClass)`
  devem ser removidas — são detalhes de implementação não contratatuais.
- A primeira chamada de `upscale()` após `load()` será mais lenta (compilação JIT lazy).
  Runs com apenas uma imagem não se beneficiam do `torch.compile`; runs com múltiplas
  imagens amortizam o custo de compilação.
- `cudnn.benchmark = True` é estado global do processo. É compatível com múltiplos runners
  instanciados simultaneamente (Fase 5 — multi-GPU).
- Runners CPU, mock e bicubic não são afetados por esta ADR.
