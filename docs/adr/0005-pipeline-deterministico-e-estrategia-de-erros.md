# ADR 0005: Pipeline Determinístico e Estratégia de Erros

## Status

Aceita

## Contexto

O sistema precisa ser operacionalmente previsível e produzir resultados comparáveis entre execuções.

## Decisão

Executar o pipeline de forma síncrona e determinística, com ordenação estável das entradas, continuidade em falhas recuperáveis por item e aborto em falhas estruturais.

## Consequências

- A ordem de processamento deve ser estável.
- Arquivos corrompidos não derrubam toda a run.
- Modelo inválido, peso ausente, device inválido ou diretório inexistente abortam cedo.
- Os comportamentos de falha devem ficar explícitos em log, status por item e manifesto.
