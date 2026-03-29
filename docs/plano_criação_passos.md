Abaixo está a decomposição do plano em passos executáveis, na ordem correta de dependência técnica e validação. A sequência foi organizada para que cada etapa produza um artefato testável antes da próxima, o que é o formato mais seguro para execução com Codex CLI ou Claude Code CLI. Não existe garantia absoluta de que nenhum ajuste será necessário durante a implementação, mas esta ordem é a mais coerente para reduzir retrabalho, isolar falhas cedo e validar o sistema de forma progressiva.

Passo 1 — Estruturar o projeto e o ambiente

Objetivo
Estabelecer a base operacional do repositório, a convenção de diretórios, os arquivos de dependência e o ponto inicial de execução do projeto.

Escopo
Criar a estrutura de pastas do projeto, os arquivos de requirements, o pacote Python principal, o README, o .gitignore e um comando inicial de entrada da aplicação.

Resultado esperado
O projeto deve abrir, instalar dependências e possuir uma CLI mínima executável, ainda que sem lógica de negócio real. A aplicação precisa ser reconhecida como um projeto Python organizado, com diretórios estáveis para código, dados, pesos, configs, runs e testes.

Como testar
Criar a venv, instalar dependências, executar o entrypoint da CLI e confirmar que o comando responde corretamente, por exemplo exibindo help ou uma mensagem padrão. O teste é bem-sucedido quando o ambiente sobe sem erro e a estrutura está pronta para evolução.

Passo 2 — Implementar a camada de configuração

Objetivo
Dar à aplicação uma forma padronizada de receber, validar e normalizar parâmetros de execução.

Escopo
Criar o carregamento de configuração por arquivo YAML e por argumentos da CLI, com precedência clara entre defaults, config e flags de execução.

Resultado esperado
A aplicação deve conseguir resolver uma configuração final consistente contendo modelo, paths, runtime, escala e parâmetros básicos. Essa configuração precisa existir como objeto interno validado e pronto para ser consumido pelo restante do pipeline.

Como testar
Executar a CLI com um arquivo de configuração válido, depois com sobrescrita por flags, e verificar se a saída final reflete corretamente a precedência esperada. Testar também arquivos inválidos, campos ausentes e parâmetros inconsistentes.

Passo 3 — Implementar descoberta e validação de entradas

Objetivo
Permitir que a aplicação identifique, filtre e organize corretamente as imagens que serão processadas.

Escopo
Criar a lógica de descoberta de arquivos suportados em diretório, validação de formatos, ordenação determinística e rejeição de entradas inválidas.

Resultado esperado
A aplicação deve conseguir montar uma lista estável de tarefas de processamento a partir de uma pasta de imagens. Arquivos inválidos não devem entrar no pipeline, mas precisam ser identificados de forma controlada.

Como testar
Executar a leitura sobre um diretório contendo imagens válidas, arquivos não suportados e arquivos corrompidos. Verificar se apenas os itens corretos entram na lista final, em ordem determinística, e se os inválidos são tratados sem derrubar a execução.

Passo 4 — Implementar a estrutura de rodada de execução

Objetivo
Transformar cada execução em uma unidade isolada, rastreável e auditável.

Escopo
Criar a lógica de geração de run_id, criação da árvore de diretórios da rodada e persistência da configuração efetiva resolvida.

Resultado esperado
Cada execução deve gerar uma pasta própria dentro de runs, contendo estrutura mínima pronta para outputs, logs, métricas e manifesto. A aplicação passa a ter uma identidade de execução formal.

Como testar
Disparar uma execução vazia ou simulada e verificar se a pasta da rodada foi criada corretamente, com nome padronizado e arquivos iniciais esperados. Repetir a execução e confirmar que novas rodadas não sobrescrevem as anteriores.

Passo 5 — Implementar logging e rastreabilidade básica

Objetivo
Registrar o comportamento da aplicação durante a execução.

Escopo
Adicionar logs em console e em arquivo, incluindo início e fim da execução, configuração efetiva, arquivos descobertos, warnings e falhas.

Resultado esperado
A aplicação deve produzir histórico técnico suficiente para depuração e auditoria. O operador precisa conseguir entender o que aconteceu em uma rodada sem olhar o código.

Como testar
Executar a aplicação com entradas válidas e inválidas e confirmar se logs relevantes são escritos tanto no terminal quanto no arquivo da rodada. Validar se mensagens essenciais aparecem nos eventos corretos.

Passo 6 — Definir a interface comum de modelos

Objetivo
Desacoplar o pipeline principal da implementação específica do primeiro modelo.

Escopo
Criar um contrato unificado para os runners de super-resolution, incluindo carregamento, inferência, descarregamento e metadados do modelo.

Resultado esperado
O restante da aplicação deve poder falar com “um modelo” sem depender do nome ou do código interno de uma implementação específica. Isso cria a base correta para adicionar outros modelos depois sem reestruturar o pipeline.

Como testar
Criar um runner mock ou dummy que implemente a interface e verificar se ele pode ser registrado e chamado pelo pipeline. O teste é bem-sucedido quando a aplicação consegue usar um modelo abstrato sem conhecimento específico de implementação.

Passo 7 — Implementar registro e seleção de modelos

Objetivo
Permitir que a aplicação escolha o backend de inferência por nome lógico.

Escopo
Criar o registro de modelos e o mecanismo de resolução do modelo solicitado na configuração.

Resultado esperado
Ao informar um nome de modelo, a aplicação deve localizar a implementação correspondente e instanciá-la corretamente. Modelos desconhecidos devem falhar cedo e de forma clara.

Como testar
Executar a aplicação com um nome de modelo válido e outro inválido. Verificar se o válido é resolvido corretamente e se o inválido gera erro controlado e compreensível.

Passo 8 — Integrar o primeiro runner real de super-resolution

Objetivo
Fazer a aplicação processar imagens reais com um modelo de verdade.

Escopo
Implementar o runner do primeiro modelo, incluindo carregamento de pesos, seleção de device, precisão e inferência sobre uma imagem.

Resultado esperado
A aplicação deve ser capaz de receber uma imagem válida, executar super-resolution e devolver uma imagem ampliada de saída. Esse é o primeiro ponto em que o sistema deixa de ser estrutural e passa a gerar valor funcional concreto.

Como testar
Executar o runner diretamente sobre uma imagem conhecida e validar que o output foi gerado, possui dimensões compatíveis com a escala desejada e não resulta em falha de execução. Repetir em CPU e, se disponível, em CUDA.

Passo 9 — Implementar o pipeline batch de inferência

Objetivo
Aplicar o modelo integrado sobre um conjunto de imagens, não apenas sobre um item isolado.

Escopo
Construir o loop principal de batch, lendo imagens, chamando o modelo, salvando outputs e registrando resultados por item.

Resultado esperado
A aplicação deve conseguir processar um diretório inteiro de imagens, preservando continuidade mesmo se um item falhar. O resultado visível é uma rodada completa com múltiplos outputs gerados.

Como testar
Executar o comando de inferência sobre uma pasta pequena de imagens válidas e verificar se todos os arquivos esperados foram processados. Depois inserir um arquivo problemático no conjunto e confirmar que a rodada continua, registrando a falha daquele item.

Passo 10 — Implementar medição de tempo e resultado por imagem

Objetivo
Tornar o pipeline mensurável em nível de item e de rodada.

Escopo
Registrar tempo de inferência por imagem, dimensões de entrada e saída, status individual e agregação parcial da execução.

Resultado esperado
Cada item processado deve produzir um registro técnico próprio, e a rodada deve começar a acumular dados de desempenho. A aplicação passa a fornecer base objetiva para comparar custo computacional entre execuções.

Como testar
Executar uma rodada pequena e verificar se cada imagem possui dados associados de tempo e status, e se esses dados aparecem corretamente no sumário parcial ou nas estruturas internas persistidas.

Passo 11 — Implementar manifesto final da rodada

Objetivo
Consolidar a execução em um artefato técnico auditável.

Escopo
Gerar o manifest.json com identificação da rodada, comando executado, configuração efetiva, metadados do modelo, parâmetros de runtime, totais de processamento e estatísticas agregadas.

Resultado esperado
Toda rodada concluída deve ter um manifesto completo, suficiente para reconstituir o contexto técnico da execução sem depender de memória ou inspeção manual do terminal.

Como testar
Executar uma rodada e validar o conteúdo do manifesto: presença de campos obrigatórios, coerência dos valores, contagem de arquivos processados, referência correta ao modelo e tempos agregados compatíveis com os resultados observados.

Passo 12 — Implementar tratamento de erro por item e erro global

Objetivo
Estabilizar o comportamento da aplicação em cenários imperfeitos.

Escopo
Separar falhas recuperáveis por imagem de falhas fatais globais, garantindo comportamento previsível e mensagens úteis.

Resultado esperado
A aplicação deve continuar quando uma imagem falhar, mas deve abortar cedo quando faltar peso, diretório, modelo ou device válidos. O sistema deixa de depender de condições ideais para operar.

Como testar
Validar pelo menos quatro cenários: arquivo corrompido, peso inexistente, modelo inválido e diretório de entrada inexistente. Confirmar que cada caso tem o comportamento esperado, sem ambiguidade.

Passo 13 — Implementar benchmark com referência

Objetivo
Medir qualidade de forma objetiva quando houver ground truth.

Escopo
Criar pareamento entre outputs e referências, calcular métricas full-reference e salvar resultados por imagem e resumo agregado.

Resultado esperado
A aplicação deve gerar arquivos de benchmark contendo métricas como PSNR, SSIM e LPIPS, permitindo avaliar fidelidade e similaridade entre saída e referência.

Como testar
Executar benchmark sobre um conjunto pareado pequeno com nomes compatíveis e verificar se as métricas são calculadas corretamente, salvas em arquivo e agregadas no sumário. Também testar a ausência de pares válidos e inconsistências de nomes.

Passo 14 — Implementar benchmark sem referência

Objetivo
Avaliar imagens reais quando não existir ground truth.

Escopo
Adicionar cálculo de métricas no-reference, como NIQE, sobre os outputs gerados.

Resultado esperado
A aplicação deve conseguir produzir avaliação quantitativa também em cenários reais, nos quais não há imagem-alvo de alta resolução para comparação direta.

Como testar
Executar benchmark sobre outputs gerados sem fornecer referência e verificar se métricas no-reference são calculadas, salvas e agregadas corretamente.

Passo 15 — Implementar comparação entre rodadas

Objetivo
Permitir análise objetiva entre diferentes execuções do sistema.

Escopo
Ler manifestos e arquivos de métricas de duas ou mais rodadas e consolidar tempos, parâmetros e resultados de qualidade.

Resultado esperado
A aplicação deve produzir uma visão comparativa entre execuções, evidenciando diferenças de modelo, runtime, qualidade e desempenho. Esse passo transforma execuções isoladas em base real de decisão técnica.

Como testar
Gerar pelo menos duas rodadas distintas, com configurações diferentes, e executar a comparação. Validar se os deltas de tempo, métricas e parâmetros aparecem corretamente e se os dados correspondem às rodadas originais.

Passo 16 — Implementar relatório consolidado

Objetivo
Expor os resultados de forma legível para leitura humana.

Escopo
Gerar um relatório consolidado em HTML simples, reunindo identificação da rodada, parâmetros, tempos, métricas e artefatos comparativos.

Resultado esperado
A aplicação deve conseguir transformar dados técnicos brutos em um artefato legível, adequado para análise rápida, revisão interna e validação visual.

Como testar
Executar a geração de relatório sobre uma rodada e sobre uma comparação entre rodadas. Validar se o HTML é gerado corretamente, abre no navegador e contém as informações esperadas.

Passo 17 — Integrar segundo modelo sob a mesma interface

Objetivo
Provar que a arquitetura suporta múltiplos backends sem reescrita do pipeline.

Escopo
Adicionar um segundo runner real, registrá-lo na aplicação e garantir que toda a camada de batch, benchmark e comparação continue funcionando sem alterações estruturais.

Resultado esperado
A aplicação deve ser capaz de trocar de modelo apenas por configuração, mantendo o mesmo fluxo operacional. Isso confirma que o desacoplamento entre pipeline e modelo foi implementado corretamente.

Como testar
Executar uma rodada com o primeiro modelo e outra com o segundo, nas mesmas imagens, e verificar se ambas completam com os mesmos mecanismos de saída, manifesto, benchmark e comparação.

Passo 18 — Consolidar testes automatizados mínimos

Objetivo
Fixar a estabilidade da base antes da expansão futura.

Escopo
Cobrir com testes automatizados os pontos mais sensíveis: parsing de config, descoberta de arquivos, criação de rodada, resolução de modelo, geração de manifesto e lógica de comparação.

Resultado esperado
O projeto deve ter uma base mínima de testes que impeça regressões óbvias ao adicionar novos modelos, novos relatórios ou mudanças no fluxo da CLI.

Como testar
Executar a suíte de testes e validar que os cenários críticos passam de forma reprodutível. O objetivo aqui é estabilização, não cobertura total.

Ordem final recomendada para execução
Estruturar projeto e ambiente
Implementar configuração
Implementar descoberta e validação de entradas
Implementar estrutura de rodada
Implementar logging
Definir interface comum de modelos
Implementar registro de modelos
Integrar primeiro runner real
Implementar pipeline batch
Implementar medição por imagem
Implementar manifesto final
Implementar tratamento de erros
Implementar benchmark com referência
Implementar benchmark sem referência
Implementar comparação entre rodadas
Implementar relatório consolidado
Integrar segundo modelo
Consolidar testes automatizados

Essa ordem respeita a dependência real entre os componentes: primeiro a base, depois o núcleo funcional, depois a mensuração, depois a avaliação, depois a comparação, depois a extensibilidade. É a sequência mais adequada para desenvolvimento incremental com validação contínua.