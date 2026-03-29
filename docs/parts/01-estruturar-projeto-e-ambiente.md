# Passo 01: Estruturar o Projeto e o Ambiente

## Objetivo

Criar a base operacional do repositório para que a aplicação seja reconhecida como um projeto Python 3.12+ executável por CLI, com organização previsível para código, dados, pesos, configurações, runs e testes.

## Decisões fixas deste passo

- Linguagem: Python 3.12+
- Interface principal: CLI
- Tipo de sistema: aplicação local monolítica
- Persistência: filesystem
- Stack mínima prevista: Typer, Rich, PyYAML, NumPy, OpenCV, Pillow, PyTorch, scikit-image, pyiqa

## Dependências

- Leitura e concordância com a especificação técnica
- Nenhum passo anterior

## Entregáveis

- Estrutura inicial de diretórios
- Pacote Python principal
- Arquivo de dependências
- `README.md`
- `.gitignore`
- Entrypoint de CLI mínimo

## Estrutura sugerida

```text
src/upscale_image/
  cli/
  config/
  io/
  models/
  pipeline/
  metrics/
  reports/
tests/
configs/
weights/
data/
runs/
```

## Implementação esperada

1. Criar o pacote principal e adotar `src layout`.
2. Definir um comando principal da CLI com Typer.
3. Garantir que o projeto instale e que a CLI responda a `--help`.
4. Preparar diretórios que serão usados pelos próximos passos, mesmo que ainda vazios.
5. Documentar no `README` o propósito técnico da aplicação e a forma mínima de execução.

## Critérios de aceite

- O ambiente virtual instala dependências sem erro.
- O comando principal da CLI executa e mostra ajuda.
- A organização de diretórios já comporta config, modelos, pipeline, métricas, relatórios e runs.
- O projeto está pronto para crescer sem refatoração estrutural imediata.

## Como testar

- Criar a venv e instalar dependências.
- Executar a CLI sem argumentos e com `--help`.
- Verificar se o pacote pode ser importado sem erro.

## Armadilhas a evitar

- Misturar código de infraestrutura com lógica de inferência já neste passo.
- Criar uma estrutura de pastas orientada a um único modelo específico.
- Deixar `runs/` e `configs/` indefinidos, pois eles são centrais para o restante do pipeline.
