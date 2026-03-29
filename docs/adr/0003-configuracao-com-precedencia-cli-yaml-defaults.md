# ADR 0003: Configuração com Precedência CLI > YAML > Defaults

## Status

Aceita

## Contexto

O sistema precisa combinar repetibilidade com flexibilidade operacional.

## Decisão

Resolver configuração usando a precedência `CLI > YAML > defaults`.

## Consequências

- Experimentos repetíveis podem ser versionados em YAML.
- Ajustes pontuais de execução podem ser feitos via CLI.
- A configuração efetiva precisa ser persistida por run.
- O merge de configuração deve ser centralizado e validado cedo.
