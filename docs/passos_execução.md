### Fluxo conceitual da aplicação (POC Super-Resolution)

1. **Receber entrada**

   * Receber diretório de imagens
   * Receber configuração (modelo, escala, parâmetros)

2. **Validar execução**

   * Verificar paths (input, output, weights)
   * Verificar modelo selecionado
   * Verificar disponibilidade de device (CPU/GPU)

3. **Preparar execução**

   * Criar identificador da rodada (run)
   * Criar estrutura de diretórios da execução
   * Registrar configuração efetiva

4. **Descobrir imagens**

   * Listar arquivos válidos
   * Filtrar formatos suportados
   * Ordenar de forma determinística

5. **Carregar modelo**

   * Inicializar modelo selecionado
   * Carregar pesos
   * Configurar runtime (device, precisão, tile)

6. **Processar imagens (loop batch)**

   * Para cada imagem:

     * Ler imagem
     * Aplicar pré-processamento (se necessário)
     * Executar super-resolution
     * Aplicar pós-processamento
     * Salvar resultado
     * Registrar tempo e status

7. **Gerenciar falhas**

   * Capturar erros por imagem
   * Continuar execução
   * Registrar falhas

8. **Finalizar execução**

   * Calcular estatísticas gerais (tempo total, média, sucesso)
   * Gerar manifesto da rodada
   * Salvar logs

9. **Executar benchmark (opcional)**

   * Parear outputs com referências
   * Calcular métricas (FR e NR)
   * Gerar métricas por imagem
   * Gerar resumo agregado

10. **Comparar execuções (opcional)**

    * Ler múltiplas rodadas
    * Comparar métricas e tempos
    * Calcular diferenças

11. **Gerar relatório (opcional)**

    * Consolidar resultados
    * Organizar evidências visuais
    * Expor métricas e comparações

12. **Encerrar**

    * Liberar recursos (modelo, memória)
    * Exibir resumo final ao usuário

---

### Resultado final esperado

* Outputs de imagem
* Manifesto da execução
* Logs
* Métricas (se aplicável)
* Relatórios (se aplicável)
