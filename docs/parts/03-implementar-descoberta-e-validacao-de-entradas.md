# Passo 03: Implementar Descoberta e Validação de Entradas

## Objetivo

Descobrir imagens de forma determinística, filtrar formatos suportados e bloquear entradas inválidas sem derrubar toda a execução.

## Extensões suportadas

```text
.png, .jpg, .jpeg, .webp, .bmp, .tif, .tiff
```

## Dependências

- Passo 02 concluído

## Entregáveis

- Função de varredura de diretório
- Filtro por extensão suportada
- Validação de leitura de imagem
- Ordenação determinística
- Estrutura de tarefa por imagem

## Estrutura mínima de task

```python
class ImageTask:
    input_path: str
    output_path: str
    filename: str
    status: str
```

## Implementação esperada

1. Receber um diretório de entrada validado.
2. Listar apenas arquivos elegíveis.
3. Verificar se o arquivo pode ser aberto de forma segura.
4. Construir uma lista estável de `ImageTask`.
5. Separar claramente itens rejeitados, warnings e itens prontos para o pipeline.

## Critérios de aceite

- A ordem final de processamento é estável entre execuções.
- Arquivos não suportados ficam fora do pipeline.
- Arquivos corrompidos não causam falha global nesta etapa.
- O pipeline recebe tarefas completas, não strings soltas de path.

## Como testar

- Diretório misto com imagens válidas e inválidas.
- Diretório vazio.
- Arquivos com extensão válida, mas conteúdo inválido.
- Diferença de ordenação entre sistemas de arquivos.

## Armadilhas a evitar

- Confiar apenas na extensão do arquivo.
- Deixar a definição de `output_path` para o meio do loop batch.
- Permitir que a ordem de processamento dependa do retorno bruto do filesystem.
