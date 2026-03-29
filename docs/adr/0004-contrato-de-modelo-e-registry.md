# ADR 0004: Contrato de Modelo e Registry

## Status

Aceita

## Contexto

O pipeline precisa suportar múltiplos backends de super-resolution sem reestruturação constante.

## Decisão

Padronizar os modelos por meio de uma interface base com `load`, `upscale` e `unload`, e resolver implementações por registry.

## Consequências

- O pipeline não conhece modelos concretos.
- Novos modelos serão adicionados implementando contrato e registry.
- Benchmark, comparação e relatório permanecem independentes do backend específico.
