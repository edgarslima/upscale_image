# Passo 19: Implementar Otimização de Imagens Geradas

## Objetivo

Reduzir o tamanho dos artefatos finais de imagem sem alterar os `outputs/*.png` que representam a saída canônica da inferência.

## ADRs relacionadas

- [0002-persistencia-orientada-a-filesystem-e-runs.md](/home/edgar/dev/upscale_image/docs/adr/0002-persistencia-orientada-a-filesystem-e-runs.md)
- [0003-configuracao-com-precedencia-cli-yaml-defaults.md](/home/edgar/dev/upscale_image/docs/adr/0003-configuracao-com-precedencia-cli-yaml-defaults.md)
- [0005-pipeline-deterministico-e-estrategia-de-erros.md](/home/edgar/dev/upscale_image/docs/adr/0005-pipeline-deterministico-e-estrategia-de-erros.md)
- [0006-observabilidade-com-logs-manifesto-e-metricas.md](/home/edgar/dev/upscale_image/docs/adr/0006-observabilidade-com-logs-manifesto-e-metricas.md)
- [0009-relatorios-e-comparacao-entre-runs.md](/home/edgar/dev/upscale_image/docs/adr/0009-relatorios-e-comparacao-entre-runs.md)
- [0010-otimizacao-derivada-e-nao-destrutiva-dos-outputs.md](/home/edgar/dev/upscale_image/docs/adr/0010-otimizacao-derivada-e-nao-destrutiva-dos-outputs.md)

## Problema a resolver

Os PNGs gerados pela inferência servem como base técnica estável, mas costumam ser pesados para distribuição. O sistema precisa gerar versões otimizadas sem sobrescrever os artefatos usados por benchmark, comparação e auditoria.

## Dependências

- Passo 09 concluído
- Passo 10 concluído
- Passo 11 concluído
- Passo 16 concluído
- Passo 18 concluído

## Entregáveis

- Camada de otimização derivada sobre uma `run` já concluída
- Configuração explícita de otimização com presets determinísticos
- Diretório `optimized/` dentro da `run`
- Resumo por imagem e resumo agregado da otimização
- Manifesto atualizado para referenciar artefatos otimizados quando existirem

## Escopo obrigatório

- Ler imagens apenas de `runs/<run_id>/outputs/*.png`
- Gerar derivados em formatos de distribuição, no mínimo `webp` e `jpeg`
- Persistir derivados sem alterar `outputs/`
- Registrar tamanho original, tamanho otimizado, economia absoluta, economia percentual, formato alvo e status
- Permitir execução repetível com a mesma configuração

## Fora de escopo

- Reexecutar inferência
- Substituir o output canônico da `run`
- Recalcular benchmark usando arquivos otimizados
- Introduzir dependência de storage externo ou banco de dados

## Estrutura de saída esperada

```text
runs/<run_id>/
  outputs/
    *.png
  optimized/
    summary.json
    per_image.csv
    webp/
      *.webp
    jpeg/
      *.jpg
```

## Implementação esperada

1. Introduzir um fluxo dedicado de otimização orientado a `run`, não ao diretório bruto de entrada.
2. Resolver a configuração de otimização com precedência `CLI > YAML > defaults`, preservando a regra da ADR 0003.
3. Validar cedo a existência de `manifest.json`, `outputs/` e ao menos um `*.png` elegível antes de iniciar processamento.
4. Processar arquivos em ordem determinística por nome, preservando o comportamento operacional da ADR 0005.
5. Usar os PNGs de `outputs/` como fonte imutável e gravar derivados apenas em `optimized/`, conforme a ADR 0010.
6. Persistir `per_image.csv` com uma linha por arquivo contendo:
   `filename`, `source_format`, `target_format`, `source_bytes`, `optimized_bytes`, `bytes_saved`, `saving_ratio`, `status`, `error`.
7. Persistir `summary.json` com totais da execução:
   quantidade elegível, quantidade otimizada, quantidade com falha, bytes originais, bytes finais, bytes economizados, percentual agregado e configuração efetiva.
8. Atualizar o manifesto da `run` para referenciar a existência de `optimized/` e do resumo de otimização sem reescrever os campos históricos da inferência.
9. Registrar logs específicos da etapa: início, preset aplicado, total de arquivos, economia agregada e falhas por item.
10. Garantir que falhas de otimização por arquivo sejam recuperáveis e não invalidem a `run` original.

## Sequência sugerida para implementação com Claude Code CLI

1. Ler as ADRs listadas neste passo antes de escrever código.
2. Mapear onde a CLI atual resolve subcomandos e decidir se a otimização entra como novo subcomando ou como fluxo explícito de pós-processamento de `run`.
3. Criar o módulo de domínio da otimização separado do pipeline de inferência para evitar acoplamento indevido.
4. Implementar serialização estável de `per_image.csv` e `summary.json`.
5. Integrar logs e atualização do manifesto.
6. Cobrir o fluxo com testes de unidade e integração.
7. Validar regressão para garantir que benchmark, comparação e relatório continuem consumindo o PNG canônico.

## Critérios de aceite

- O sistema gera derivados otimizados sem alterar `outputs/*.png`.
- A etapa produz artefatos estruturados suficientes para auditoria e comparação futura.
- A execução é determinística para a mesma `run` e a mesma configuração.
- Falhas por item não corrompem a `run` original nem impedem otimização dos demais arquivos.
- O manifesto passa a apontar para os artefatos de otimização quando a etapa for executada.

## Como testar

- Executar otimização sobre uma `run` com múltiplos PNGs válidos e verificar geração em `optimized/webp/` e `optimized/jpeg/`.
- Reexecutar a etapa com a mesma configuração e confirmar comportamento determinístico.
- Incluir um PNG inválido ou ilegível dentro de `outputs/` e validar falha recuperável por item.
- Confirmar que benchmark, comparação e relatório existentes continuam lendo `outputs/*.png`.
- Validar que `summary.json` e `per_image.csv` refletem corretamente os tamanhos em disco.

## Armadilhas a evitar

- Sobrescrever `outputs/*.png` em nome de economia de espaço.
- Misturar no mesmo diretório artefatos canônicos e derivados.
- Acoplar a otimização ao runner de modelo.
- Salvar apenas os arquivos otimizados sem resumo estruturado.
- Tratar a otimização como sucesso global quando houver falhas silenciosas por item.
