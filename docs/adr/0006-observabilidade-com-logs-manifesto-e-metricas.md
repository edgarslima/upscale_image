# ADR 0006: Observabilidade com Logs, Manifesto e Métricas

## Status

Aceita

## Contexto

O valor do sistema não está apenas em gerar imagens, mas em registrar quando, como e com qual custo o processamento ocorreu.

## Decisão

Toda run deve produzir logs persistidos, configuração efetiva, manifesto técnico e, quando aplicável, métricas por imagem e agregadas.

## Consequências

- A run se torna auditável sem depender do terminal.
- Comparação entre execuções passa a ser baseada em artefatos persistidos.
- Estruturas de saída precisam ser estáveis para consumo por módulos posteriores.
