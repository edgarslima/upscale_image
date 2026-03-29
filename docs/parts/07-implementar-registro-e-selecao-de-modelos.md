# Passo 07: Implementar Registro e Seleção de Modelos

## Objetivo

Permitir que a configuração escolha o backend de inferência por nome lógico, sem condicional espalhada no código.

## Dependências

- Passo 06 concluído

## Entregáveis

- Registry de modelos
- Resolução por `model.name`
- Tratamento de modelo desconhecido

## Implementação esperada

1. Manter um mapa entre nome lógico e classe runner.
2. Receber o nome do modelo a partir da configuração resolvida.
3. Instanciar o runner correspondente.
4. Falhar cedo se o nome não existir.

## Critérios de aceite

- Um modelo válido é resolvido sem ambiguidade.
- Um modelo inválido gera erro fatal claro.
- O pipeline não usa `if/else` por modelo.

## Como testar

- Nome de modelo válido.
- Nome inválido.
- Registro duplicado.

## Armadilhas a evitar

- Registrar modelos apenas por efeito colateral de import.
- Misturar resolução de modelo com parsing de CLI.
- Permitir nomes inconsistentes entre configuração, manifesto e logs.
