# ADR 0008: Avaliação de Qualidade Full-Reference e No-Reference

## Status

Aceita

## Contexto

A aplicação precisa avaliar qualidade tanto em datasets pareados quanto em cenários reais sem ground truth.

## Decisão

Suportar dois modos analíticos:

- Full-reference com PSNR, SSIM e LPIPS
- No-reference com NIQE

## Consequências

- O sistema deve distinguir explicitamente o modo de benchmark executado.
- As métricas devem ser persistidas por imagem e em resumo agregado.
- Comparação entre runs pode considerar qualidade objetiva e não apenas tempo.
