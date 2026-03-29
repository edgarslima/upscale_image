# Passo 02: Implementar a Camada de Configuração

## Objetivo

Estabelecer uma forma única, validada e previsível de resolver parâmetros de execução a partir de defaults, YAML e flags da CLI.

## Decisão técnica obrigatória

A precedência de configuração é:

```text
CLI > YAML > defaults
```

## Dependências

- Passo 01 concluído

## Entregáveis

- Estrutura de configuração interna
- Loader de YAML
- Merge com argumentos da CLI
- Validação de campos obrigatórios e compatibilidade de parâmetros
- Persistência da configuração efetiva para uso posterior na run

## Campos mínimos esperados

```yaml
model:
  name: realesrgan
  scale: 4

runtime:
  device: cuda
  precision: fp16
  tile_size: 0
  tile_pad: 10
```

## Implementação esperada

1. Definir um objeto de configuração canônico que o restante do sistema consuma.
2. Ler YAML com PyYAML.
3. Receber sobrescritas pela CLI.
4. Normalizar paths, escala, device, precisão e parâmetros de runtime.
5. Falhar cedo quando a configuração estiver incompleta ou inconsistente.

## Critérios de aceite

- A aplicação consegue resolver a configuração final de forma determinística.
- Flags da CLI realmente sobrescrevem YAML.
- Valores inválidos geram erro claro antes do pipeline começar.
- O objeto de configuração final é consumível pelos próximos módulos sem parsing adicional.

## Como testar

- Testar defaults puros.
- Testar YAML válido.
- Testar YAML mais flags conflitantes.
- Testar device inválido, escala inválida, caminhos ausentes e combinações inconsistentes.

## Armadilhas a evitar

- Espalhar leitura de config em múltiplos pontos do sistema.
- Misturar parsing com regras de negócio do pipeline.
- Permitir que módulos downstream interpretem flags brutas por conta própria.
