# Passo 13: Implementar Benchmark com Referência

## Objetivo

Calcular métricas full-reference para medir fidelidade quando houver ground truth.

## Métricas obrigatórias

- PSNR
- SSIM
- LPIPS

## Dependências

- Passo 09 concluído
- Passo 10 concluído
- Passo 11 concluído

## Entregáveis

- Pareamento entre output e referência
- Métricas por imagem
- Resumo agregado
- Persistência em `metrics/`

## Estrutura de saída esperada

```text
metrics/
  per_image.csv
  summary.json
```

## Implementação esperada

1. Definir regra estável de pareamento por nome.
2. Validar ausência ou inconsistência de pares.
3. Calcular PSNR e SSIM com `scikit-image`.
4. Calcular LPIPS via `pyiqa`.
5. Salvar resultados de forma consumível por comparação e relatório.

## Critérios de aceite

- O benchmark funciona em dataset pareado.
- O sistema diferencia claramente ausência de par e falha de métrica.
- O resumo agregado reflete os resultados por imagem.

## Como testar

- Pequeno conjunto pareado válido.
- Conjunto com pares ausentes.
- Nomes inconsistentes entre saída e referência.

## Armadilhas a evitar

- Calcular benchmark dentro do loop de inferência.
- Acoplar métricas a um modelo específico.
- Persistir apenas resumo agregado sem dados por imagem.
