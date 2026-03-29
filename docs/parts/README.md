# Partes de Implementação

Este diretório divide o plano de criação em passos operacionais independentes, mas encadeados. O objetivo é servir como guia de execução para agentes como Codex CLI e Claude Code CLI durante a implementação da aplicação de super-resolution batch.

Os arquivos foram escritos com base em:

- `docs/plano_criação_passos.md`
- `docs/plano_criação.md`
- `docs/propósito_aplicação.md`
- `docs/passos_execução.md`
- `docs/especificação_tecnica.md`

## Como usar esta pasta

1. Execute os passos na ordem.
2. Considere cada passo concluído apenas quando os critérios de aceite e testes estiverem satisfeitos.
3. Não antecipe decisões que já estão fixadas na especificação técnica ou nas ADRs.
4. Se um passo exigir refatoração do anterior, preserve compatibilidade com os artefatos já definidos: CLI, run, manifesto, logs, métricas e relatórios.

## Princípios que orientam todos os passos

- A aplicação é local, monolítica e orientada a CLI.
- O pipeline é síncrono, determinístico e baseado em filesystem.
- Cada execução deve gerar uma `run` isolada e auditável.
- A configuração deve respeitar a precedência `CLI > YAML > defaults`.
- O pipeline não pode depender de um modelo concreto; ele depende de contrato e registry.
- Erros por item devem ser recuperáveis; erros estruturais devem abortar cedo.
- Benchmark, comparação e relatório são extensões do pipeline principal, não fluxos paralelos desconectados.

## Índice

1. [01-estruturar-projeto-e-ambiente.md](/home/edgar/dev/upscale_image/docs/parts/01-estruturar-projeto-e-ambiente.md)
2. [02-implementar-camada-de-configuracao.md](/home/edgar/dev/upscale_image/docs/parts/02-implementar-camada-de-configuracao.md)
3. [03-implementar-descoberta-e-validacao-de-entradas.md](/home/edgar/dev/upscale_image/docs/parts/03-implementar-descoberta-e-validacao-de-entradas.md)
4. [04-implementar-estrutura-de-rodada.md](/home/edgar/dev/upscale_image/docs/parts/04-implementar-estrutura-de-rodada.md)
5. [05-implementar-logging-e-rastreabilidade-basica.md](/home/edgar/dev/upscale_image/docs/parts/05-implementar-logging-e-rastreabilidade-basica.md)
6. [06-definir-interface-comum-de-modelos.md](/home/edgar/dev/upscale_image/docs/parts/06-definir-interface-comum-de-modelos.md)
7. [07-implementar-registro-e-selecao-de-modelos.md](/home/edgar/dev/upscale_image/docs/parts/07-implementar-registro-e-selecao-de-modelos.md)
8. [08-integrar-primeiro-runner-real.md](/home/edgar/dev/upscale_image/docs/parts/08-integrar-primeiro-runner-real.md)
9. [09-implementar-pipeline-batch-de-inferencia.md](/home/edgar/dev/upscale_image/docs/parts/09-implementar-pipeline-batch-de-inferencia.md)
10. [10-implementar-medicao-de-tempo-e-resultado-por-imagem.md](/home/edgar/dev/upscale_image/docs/parts/10-implementar-medicao-de-tempo-e-resultado-por-imagem.md)
11. [11-implementar-manifesto-final-da-rodada.md](/home/edgar/dev/upscale_image/docs/parts/11-implementar-manifesto-final-da-rodada.md)
12. [12-implementar-tratamento-de-erros.md](/home/edgar/dev/upscale_image/docs/parts/12-implementar-tratamento-de-erros.md)
13. [13-implementar-benchmark-com-referencia.md](/home/edgar/dev/upscale_image/docs/parts/13-implementar-benchmark-com-referencia.md)
14. [14-implementar-benchmark-sem-referencia.md](/home/edgar/dev/upscale_image/docs/parts/14-implementar-benchmark-sem-referencia.md)
15. [15-implementar-comparacao-entre-rodadas.md](/home/edgar/dev/upscale_image/docs/parts/15-implementar-comparacao-entre-rodadas.md)
16. [16-implementar-relatorio-consolidado.md](/home/edgar/dev/upscale_image/docs/parts/16-implementar-relatorio-consolidado.md)
17. [17-integrar-segundo-modelo.md](/home/edgar/dev/upscale_image/docs/parts/17-integrar-segundo-modelo.md)
18. [18-consolidar-testes-automatizados.md](/home/edgar/dev/upscale_image/docs/parts/18-consolidar-testes-automatizados.md)

## Encadeamento recomendado

- Base do sistema: passos 1 a 5
- Arquitetura de modelos: passos 6 a 8
- Pipeline operacional: passos 9 a 12
- Camada analítica: passos 13 a 16
- Extensibilidade e estabilização: passos 17 e 18

## Resultado esperado ao final da pasta

Ao concluir todos os arquivos deste diretório, a aplicação deve ser capaz de:

- executar super-resolution em lote;
- persistir cada execução como uma `run` auditável;
- medir tempo, status e parâmetros efetivos;
- calcular métricas de qualidade com e sem referência;
- comparar execuções;
- gerar relatórios legíveis;
- permitir adição de novos modelos sem reestruturar o pipeline.
