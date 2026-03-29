# Passo 04: Implementar a Estrutura de Rodada

## Objetivo

Transformar cada execução em uma entidade isolada, rastreável e não sobrescrevível.

## Decisão técnica obrigatória

Cada execução deve ser persistida em:

```text
runs/<run_id>/
```

Formato de identificação:

```text
run_<timestamp>_<model>_<scale>
```

## Dependências

- Passo 02 concluído
- Passo 03 concluído

## Entregáveis

- Geração de `run_id`
- Criação de diretório da rodada
- Subpastas mínimas da run
- Persistência de `effective_config.yaml`

## Estrutura mínima esperada

```text
runs/<run_id>/
  outputs/
  metrics/
  manifest.json
  logs.txt
  effective_config.yaml
```

## Implementação esperada

1. Gerar `run_id` antes do início do processamento.
2. Criar toda a árvore da execução de forma atômica e previsível.
3. Salvar a configuração efetiva resolvida.
4. Preparar caminhos que serão reutilizados por logging, métricas e relatório.

## Critérios de aceite

- Cada execução cria uma nova pasta.
- Não há sobrescrita de runs anteriores.
- O estado mínimo da execução existe antes do loop batch começar.
- A configuração efetiva pode ser auditada depois sem depender da CLI usada.

## Como testar

- Executar duas vezes com a mesma config e verificar runs distintas.
- Validar existência das pastas e arquivos mínimos.
- Confirmar que a configuração salva corresponde ao merge final.

## Armadilhas a evitar

- Criar outputs soltos fora da pasta da run.
- Deixar geração de `run_id` ambígua ou dependente de estado global.
- Persistir só parte da configuração resolvida.
