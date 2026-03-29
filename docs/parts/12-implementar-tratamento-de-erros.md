# Passo 12: Implementar Tratamento de Erros

## Objetivo

Formalizar a estratégia de falha do sistema para manter previsibilidade operacional.

## Decisão técnica obrigatória

- Erro recuperável por item: registrar e continuar.
- Erro fatal estrutural: abortar a execução.

## Dependências

- Passo 09 concluído
- Passo 11 concluído

## Casos mínimos a cobrir

- imagem corrompida
- erro de leitura
- falha isolada de inferência
- modelo não encontrado
- peso inexistente
- device inválido
- diretório de entrada inexistente

## Implementação esperada

1. Classificar exceções por categoria.
2. Garantir mensagens claras ao operador.
3. Refletir o tipo de falha em logs, resultado por item e status final da run.
4. Evitar abortos silenciosos ou continuação indevida em erro fatal.

## Critérios de aceite

- O comportamento do sistema é previsível em cenários imperfeitos.
- A rodada continua apenas quando faz sentido técnico continuar.
- O operador entende rapidamente a causa da falha.

## Como testar

- Reproduzir cada cenário mínimo documentado.
- Validar comportamento da CLI, logs e manifesto.

## Armadilhas a evitar

- Tratar toda exceção como warning.
- Engolir traceback útil sem registrar contexto.
- Permitir que uma run fatal pareça concluída com sucesso.
