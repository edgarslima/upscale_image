# Passo 09: Implementar o Pipeline Batch de Inferência

## Objetivo

Aplicar o modelo integrado sobre um conjunto de imagens, preservando continuidade da rodada e consistência de artefatos.

## Dependências

- Passo 03 concluído
- Passo 04 concluído
- Passo 05 concluído
- Passo 08 concluído

## Entregáveis

- Loop principal de processamento
- Leitura por item
- Chamada do runner
- Escrita de output em `outputs/`
- Registro de status individual

## Fluxo por item

1. Ler a imagem.
2. Validar se a leitura foi bem-sucedida.
3. Executar inferência.
4. Salvar o output com nome consistente ao input.
5. Registrar sucesso ou falha.

## Implementação esperada

1. Processar as tarefas na ordem descoberta.
2. Isolar exceções por item.
3. Continuar a rodada quando houver falhas recuperáveis.
4. Preparar dados que serão usados por métricas e manifesto.

## Critérios de aceite

- Um diretório de imagens é processado do início ao fim.
- A saída fica centralizada na run atual.
- Um erro por item não interrompe a rodada inteira.

## Como testar

- Pasta pequena só com imagens válidas.
- Pasta com ao menos um arquivo corrompido.
- Confirmação de nomes e quantidade de outputs.

## Armadilhas a evitar

- Misturar benchmarking no mesmo loop inicial.
- Não isolar o estado de cada item.
- Salvar outputs fora da convenção da run.
