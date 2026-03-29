# ADR 0010: Otimização Derivada e Não Destrutiva dos Outputs

## Status

Aceita

## Contexto

Os `outputs/*.png` gerados pela inferência são adequados como artefato canônico para benchmark, comparação e auditoria. Ainda assim, eles não são necessariamente o melhor formato para distribuição, armazenamento otimizado ou publicação. A aplicação precisa reduzir tamanho de artefatos quando solicitado, sem invalidar métricas já calculadas nem alterar a fonte de verdade da `run`.

## Decisão

Implementar a otimização como uma etapa derivada, executada sobre os PNGs canônicos já persistidos, gerando novos artefatos em `runs/<run_id>/optimized/` sem sobrescrever `runs/<run_id>/outputs/`.

## Consequências

- `outputs/*.png` continuam sendo a base para benchmark, comparação e reprodutibilidade.
- A otimização passa a ser auditável por arquivos próprios, resumo estruturado e referência no manifesto.
- Falhas de otimização por item não invalidam a `run` original; devem ser tratadas como falhas recuperáveis da etapa derivada.
- A economia de tamanho e os formatos derivados podem ser comparados entre `runs` sem confundir a origem do artefato.
