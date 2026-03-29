# ADR 0005: Pipeline Determinístico e Estratégia de Erros

## Status

Aceita

## Contexto

O sistema precisa ser operacionalmente previsível e produzir resultados comparáveis entre execuções.

## Decisão

Executar o pipeline com ordenação estável das entradas, continuidade em falhas recuperáveis
por item e aborto em falhas estruturais.

O determinismo é preservado em dois níveis distintos:

- **Determinismo de ordem**: a lista de tarefas é sempre produzida na mesma ordem estável
  por `discover_images()` (ordenação lexicográfica insensível a maiúsculas). O `BatchResult`
  e o manifesto listam os resultados nessa mesma ordem, independentemente do modo de execução.

- **Determinismo de execução**: o pipeline serial (padrão) processa imagens estritamente
  uma a uma. O pipeline assíncrono (ADR 0013, opt-in) permite que I/O ocorra em threads
  paralelas à inferência GPU, relaxando o determinismo de execução sem alterar o
  determinismo de ordem dos resultados.

## Consequências

- A ordem de processamento é estável em todos os modos (serial e assíncrono).
- Arquivos corrompidos não derrubam toda a run.
- Modelo inválido, peso ausente, device inválido ou diretório inexistente abortam cedo.
- Os comportamentos de falha ficam explícitos em log, status por item e manifesto.
- O modo assíncrono (ADR 0013) é opt-in. O comportamento serial padrão não muda.
- Falhas em threads de I/O no modo assíncrono são capturadas e convertidas em
  `ItemResult(status="failed")`, preservando a semântica de erro por item desta ADR.
