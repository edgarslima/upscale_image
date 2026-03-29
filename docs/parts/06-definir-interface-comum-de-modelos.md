# Passo 06: Definir a Interface Comum de Modelos

## Objetivo

Desacoplar o pipeline principal de implementações concretas de super-resolution.

## Decisão técnica obrigatória

Todo modelo integrado deve obedecer ao contrato base:

```python
class SuperResolutionModel:
    def load(self): ...
    def upscale(self, image, config): ...
    def unload(self): ...
```

## Dependências

- Passo 01 concluído
- Passo 02 concluído

## Entregáveis

- Interface ou classe abstrata de modelo
- Metadados mínimos do runner
- Runner mock para validação do contrato

## Implementação esperada

1. Definir a API pública que o pipeline pode chamar.
2. Incluir capacidade de carregar, inferir e liberar recursos.
3. Expor metadados úteis para manifesto e logs.
4. Criar um runner fake para validar o acoplamento correto.

## Critérios de aceite

- O pipeline consegue trabalhar com um modelo abstrato.
- Nenhuma lógica de batch depende de um runner específico.
- O contrato já contempla carga e liberação de recursos.

## Como testar

- Registrar e executar um runner dummy.
- Confirmar que o pipeline chama apenas a interface base.
- Validar que o runner pode ser descarregado ao final.

## Armadilhas a evitar

- Acoplar paths de peso, nomes de arquivo ou detalhes de tensor à interface base.
- Criar uma interface ampla demais sem necessidade do pipeline atual.
- Misturar responsabilidades de registry dentro do contrato base.
