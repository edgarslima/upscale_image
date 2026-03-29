# ADR 0001: Arquitetura CLI Local Monolítica

## Status

Aceita

## Contexto

O projeto tem como propósito executar super-resolution de forma controlada, mensurável e comparável em ambiente local. O fluxo principal é batch, síncrono e orientado a arquivo em disco.

## Decisão

Adotar uma aplicação local monolítica com interface principal em CLI.

## Consequências

- Simplifica a operação inicial e reduz custo arquitetural.
- Favorece previsibilidade e reprodutibilidade em ambiente de experimentação técnica.
- Não haverá, na fase inicial, API, banco de dados, streaming ou execução distribuída.
