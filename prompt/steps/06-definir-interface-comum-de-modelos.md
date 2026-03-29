# Prompt do Passo 06

Execute o passo 06 do projeto: definir a interface comum de modelos.

## Leia antes de implementar

- `/home/edgar/dev/upscale_image/docs/parts/06-definir-interface-comum-de-modelos.md`
- `/home/edgar/dev/upscale_image/docs/especificação_tecnica.md`
- `/home/edgar/dev/upscale_image/docs/adr/0004-contrato-de-modelo-e-registry.md`

## Regras fixas

- o pipeline não conhece modelo concreto;
- o contrato base deve cobrir `load`, `upscale` e `unload`;
- crie um runner mock para validar o desacoplamento.

## Antes de finalizar

- validar que o pipeline consegue conversar com o mock pelo contrato;
- atualizar `prompt/control.yaml`.
