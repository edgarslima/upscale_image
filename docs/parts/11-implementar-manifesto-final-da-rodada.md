# Passo 11: Implementar o Manifesto Final da Rodada

## Objetivo

Consolidar cada run em um artefato técnico capaz de reconstituir o contexto da execução.

## Decisão técnica obrigatória

O manifesto é persistido como `manifest.json` dentro da run.

## Dependências

- Passo 04 concluído
- Passo 10 concluído

## Estrutura mínima esperada

```json
{
  "run_id": "...",
  "model": {},
  "runtime": {},
  "timing": {},
  "status": {}
}
```

## Entregáveis

- Manifesto JSON
- Identidade da run
- Metadados do modelo
- Parâmetros de runtime
- Estatísticas agregadas
- Referência à configuração efetiva e aos artefatos gerados

## Implementação esperada

1. Consolidar dados de configuração, runtime, resultados e tempos.
2. Incluir comando executado e, quando disponível, versão ou commit do código.
3. Salvar apenas campos estáveis e úteis para auditoria e comparação.

## Critérios de aceite

- O manifesto permite entender a execução sem consultar o terminal.
- Totais e contagens batem com a rodada real.
- Os campos principais têm nomes estáveis para consumo posterior.

## Como testar

- Executar uma rodada e validar manualmente o JSON.
- Conferir coerência entre log, outputs e manifesto.

## Armadilhas a evitar

- Transformar o manifesto em dump bruto de objetos internos.
- Não preservar distinção entre configuração pedida e contexto realmente usado.
- Mudar nomes de campos sem tratar compatibilidade futura.
