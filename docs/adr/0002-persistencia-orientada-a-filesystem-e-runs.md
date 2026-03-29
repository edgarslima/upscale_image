# ADR 0002: Persistência Orientada a Filesystem e Runs

## Status

Aceita

## Contexto

Cada execução precisa ser auditável, reproduzível e comparável sem depender de infraestrutura adicional.

## Decisão

Persistir todos os artefatos em filesystem, encapsulando cada execução em `runs/<run_id>/`.

## Consequências

- Cada run terá isolamento claro.
- Logs, manifesto, métricas e outputs ficam co-localizados.
- O filesystem passa a ser a fonte primária de verdade da execução.
- O projeto evita dependência inicial de banco ou serviços externos.
