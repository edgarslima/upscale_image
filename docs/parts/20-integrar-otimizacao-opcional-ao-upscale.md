# Passo 20: Integrar Otimização Opcional ao Upscale

## Objetivo

Permitir que o comando principal de `upscale` execute a otimização de imagens automaticamente ao final da inferência quando um parâmetro opcional for informado.

## ADRs relacionadas

- [0003-configuracao-com-precedencia-cli-yaml-defaults.md](/home/edgar/dev/upscale_image/docs/adr/0003-configuracao-com-precedencia-cli-yaml-defaults.md)
- [0005-pipeline-deterministico-e-estrategia-de-erros.md](/home/edgar/dev/upscale_image/docs/adr/0005-pipeline-deterministico-e-estrategia-de-erros.md)
- [0006-observabilidade-com-logs-manifesto-e-metricas.md](/home/edgar/dev/upscale_image/docs/adr/0006-observabilidade-com-logs-manifesto-e-metricas.md)
- [0010-otimizacao-derivada-e-nao-destrutiva-dos-outputs.md](/home/edgar/dev/upscale_image/docs/adr/0010-otimizacao-derivada-e-nao-destrutiva-dos-outputs.md)

## Problema a resolver

Após o passo 19, a aplicação já consegue otimizar imagens de uma `run` concluída, mas esse fluxo ainda depende de uma ação separada. O objetivo agora é encadear essa capacidade ao comando de `upscale` por um parâmetro opcional, reduzindo atrito operacional sem alterar o comportamento padrão da inferência.

## Dependências

- Passo 09 concluído
- Passo 11 concluído
- Passo 19 concluído

## Entregáveis

- Novo parâmetro opcional no comando de `upscale`
- Encadeamento do fluxo de otimização ao final da `run`
- Reaproveitamento integral da camada do passo 19
- Persistência e manifesto idênticos aos do fluxo dedicado
- Cobertura de testes do comportamento com e sem otimização

## Escopo obrigatório

- Adicionar parâmetro opcional claro na CLI de `upscale` para habilitar otimização
- Manter a funcionalidade desabilitada por padrão
- Executar otimização somente após a inferência terminar e os `outputs/*.png` existirem
- Permitir configurar formatos ou preset de otimização a partir da CLI ou config
- Registrar no log da `run` quando a otimização encadeada foi solicitada, iniciada, concluída ou parcialmente falhou

## Fora de escopo

- Tornar otimização obrigatória em toda execução
- Reimplementar a lógica de otimização já criada no passo 19
- Alterar benchmark para consumir artefatos otimizados
- Mudar o contrato canônico de `outputs/*.png`

## Implementação esperada

1. Estender a interface da CLI de `upscale` com um parâmetro opcional explícito para ativar otimização ao final do fluxo.
2. Resolver a configuração desse parâmetro usando a precedência `CLI > YAML > defaults`, alinhada à ADR 0003.
3. Reutilizar a mesma camada de domínio do passo 19 em vez de duplicar a implementação dentro do pipeline.
4. Executar a otimização somente depois de a inferência e a escrita do manifesto principal terem sido concluídas com consistência.
5. Se a inferência falhar estruturalmente antes de gerar a `run`, não iniciar otimização.
6. Se a inferência concluir e a otimização falhar parcialmente por item, preservar a `run` como válida e registrar o status derivado conforme a ADR 0010.
7. Garantir que o caminho sem o parâmetro opcional mantenha exatamente o comportamento anterior do comando `upscale`.
8. Garantir que o resumo final da CLI deixe claro quando houve otimização encadeada e qual foi o resultado agregado.

## Sequência sugerida para implementação com Claude Code CLI

1. Ler o passo 19 e confirmar quais funções, tipos e contratos devem ser reutilizados.
2. Mapear o ponto exato da CLI onde a `run` é encerrada e inserir o gancho opcional após a produção dos artefatos canônicos.
3. Adicionar o novo parâmetro na CLI e, se aplicável, no objeto de configuração.
4. Encadear a execução da otimização sem alterar a ordem principal do pipeline.
5. Ajustar logs, manifesto e retorno final da CLI.
6. Criar testes para:
   sem parâmetro opcional, com parâmetro opcional, falha parcial na otimização e preservação dos `outputs/*.png`.

## Critérios de aceite

- O comando `upscale` continua funcionando exatamente como antes quando o novo parâmetro não é enviado.
- Quando o parâmetro é enviado, a aplicação executa a otimização automaticamente ao final da inferência.
- Os arquivos canônicos em `outputs/*.png` continuam intactos.
- Os artefatos em `optimized/` são os mesmos do fluxo dedicado do passo 19.
- Logs e manifesto deixam explícito que houve otimização encadeada.

## Como testar

- Executar `upscale` sem o novo parâmetro e validar ausência de `optimized/`.
- Executar `upscale` com o novo parâmetro e validar geração de `optimized/`, `summary.json` e `per_image.csv`.
- Comparar o resultado do `upscale` com otimização encadeada com o resultado do fluxo dedicado do passo 19 sobre a mesma `run`.
- Simular falha de otimização por item e validar que a `run` principal continua utilizável.

## Armadilhas a evitar

- Duplicar no comando `upscale` a implementação do passo 19.
- Executar otimização antes da persistência dos artefatos canônicos.
- Mudar o comportamento padrão do comando sem parâmetro opcional.
- Esconder falhas de otimização dentro de um sucesso genérico da inferência.
