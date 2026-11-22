# Auditoria técnica da Etapa 3 — NSL-KDD

Data: 22/11/2026 (America/Sao_Paulo). Escopo: Fase 1, diagnóstico e planejamento, sem implementar a consolidação dos experimentos.

## 1. Base auditada e limites das evidências

- Repositório: `JuliaSebastiao/nsl-kdd-intrusion-detection`.
- Referências remotas atualizadas com `git fetch --all --prune`; árvore de trabalho inicialmente limpa.
- Branch criada: `feat/model-evaluation-stage3`, a partir de `origin/feat/preprocessamento-etapa2`.
- Commit-base auditado: `1c907e4fa0a0db96890d8716e31d98c91ab2655a`.
- `main` local preservada em `f8f86a2fe35bd398de40ca316b651a00fa6d361b`; referência de origem preservada no commit-base. Sem merge ou push.
- Único arquivo de projeto criado nesta fase: `docs/STAGE3_AUDIT.md`. Nenhum arquivo existente alterado.

Esta auditoria combina leitura integral dos fontes, inspeção das células dos oito notebooks, comparação estrutural do código compartilhado, testes existentes, verificações sintéticas e conferência aritmética de números documentados. Referências a notebooks usam o campo `id` da célula e o nome da função, evitando depender da numeração de linhas do JSON.

Não há `treino.csv`, `teste_sem_rotulo.csv`, `modelo_de_submissao.csv` ou resultados locais de treinamentos neste checkout, inclusive entre os arquivos ignorados pesquisados. Os oito notebooks estão sem saídas e sem contadores de execução: **nenhuma métrica sobre a base oficial foi reproduzida nesta auditoria**. A ausência dos CSVs no Git é correta; não se recomenda publicá-los. Os valores de `DECISOES.md` são registros históricos, não resultados certificados por esta execução.

Não foram executados treinamentos completos, PyTorch ou cuML. Não foram acessados dados externos para substituir a base oficial. O enunciado original e os materiais da disciplina citados não estão versionados; a cobertura de requisitos foi conferida contra a solicitação desta auditoria e o registro do projeto, sem presumir regras adicionais.

## 2. Arquitetura e componentes examinados

| Arquivo/componente | Responsabilidade e verificação |
| --- | --- |
| `README.md` | Instruções, escopo das etapas, dependências e links; identificado link local de PDF inexistente. |
| `DECISOES.md` | Decisões de atributos, desenho de validação, tabelas de resultados e histórico de mudanças das redes. |
| `preprocessamento.py` | Diagnóstico da Etapa 1: leitura, tipos, domínio, duplicatas/conflitos, associações e redundâncias. Não contém o pipeline de treino das Etapas 2/3. |
| `01_analise_exploratoria.ipynb` | Análise modular; não ajusta classificadores nem modifica o CSV. |
| `01_analise_exploratoria_colab.ipynb` | Cópia independente da EDA; módulo incorporado e nove células posteriores de análise coincidem com as fontes. |
| `preparar_entrega_etapa1.py` | Gera cópias com funções incorporadas e remove saídas; geração independente verificada em memória, sem gravar notebooks. |
| `02_preprocessamento_colab.ipynb` | Comparações de representações, limpeza, imputação, codificação, escala, amostragem e redução; árvore e MLP de referência. |
| `03_random_forest_colab.ipynb` | Busca de quatro configurações e avaliação da selecionada na quinta divisão. |
| `03_redes_neurais_colab.ipynb` | Duas MLPs com 120 épocas fixas, oversampling e avaliação em cinco divisões. |
| `03_deep_learning_colab.ipynb` | Rede tabular PyTorch, quatro camadas ocultas, dropout, Adam, CPU/CUDA. |
| `03_svm_colab.ipynb` | Sete configurações LinearSVC/SVC, convergência, tempos e avaliação na quinta divisão. |
| `03_svm_cuda_colab.ipynb` | Benchmark RBF CPU/cuML, duas tolerâncias e matrizes compartilhadas; não avalia a quinta divisão. |
| `requirements.txt` | Quatro dependências da EDA; não é manifesto completo das Etapas 2/3. |
| `tests/test_preprocessamento.py` | Dez testes unitários de diagnóstico; não cobre os módulos de experimentos embutidos. |
| `.gitignore` e `.gitattributes` | Dados e temporários ignorados; PDFs tratados como binários. |
| `output/pdf/etapa1_resumo.pdf` e `output/pdf/etapa2_resumo.pdf` | Conteúdo textual das duas páginas e destinos das anotações de links inspecionados. Resumos compatíveis com o registro das respectivas etapas; não constituem evidência de execução da Etapa 3. Não houve auditoria visual de diagramação. |

Os notebooks de experimentos criam módulos temporários com `%%writefile`, instalam/verificam dependências para subprocessos e executam esses módulos. As funções `preparar_base`, `escolher_linhas`, `separar_divisoes`, `transformar_numericas`, `transformar_categoricas` e `transformar` têm a mesma árvore sintática nas seis cópias das Etapas 2/3. Isso sustenta a comparabilidade do código atual, mas a duplicação exige disciplina de manutenção.

## 3. Protocolo de validação e vazamento

### O que está correto no código atual

Nas células `definicoes-dados`, `preparar_base` exclui `class` e as quatro colunas aprovadas, normaliza categorias com `strip/casefold`, transforma violações de domínio em ausências e constrói as versões A/B/C. Os grupos são definidos por igualdade das 40 entradas de C, inclusive o padrão de ausência, sem usar o rótulo na chave. Entradas iguais com rótulos conflitantes ficam no mesmo grupo.

Essas regras anteriores à divisão são determinísticas: não estimam mediana, vocabulário, média, variância ou parâmetros com toda a base. Construir grupos com as entradas disponíveis e usar o alvo para estratificação não equivale, por si só, a vazamento. A estratificação tenta preservar as proporções sob a restrição de grupos indivisíveis; não garante proporções idênticas. [Referência: StratifiedGroupKFold 1.7](https://scikit-learn.org/1.7/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html).

`separar_divisoes` usa `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`. Define `reserva = particoes[-1][1]` e, nas primeiras quatro divisões, executa `treino = np.setdiff1d(treino, reserva)`. Há assertions de disjunção entre grupos de ajuste, validação e reserva. Portanto:

- Cada uma das quatro primeiras rodadas treina em três blocos, aproximadamente 60% do total, e valida em um bloco, aproximadamente 20%.
- A quinta rodada treina na união dos quatro primeiros blocos, aproximadamente 80%, e valida no quinto.
- Não se trata de cinco avaliações equivalentes para calcular uma única média. Os notebooks atuais corretamente separam a média das primeiras quatro do resultado da quinta.
- A identidade das divisões entre arquivos depende também do mesmo CSV, ordem de linhas e versões; a semente isolada não comprova isso para execuções históricas.

Em `transformar`, a seleção/deduplicação ocorre apenas nos índices de treino. `transformar_numericas` ajusta imputadores e normalizadores no treino; o KNN também escolhe suas até 2.000 referências exclusivamente ali. `transformar_categoricas` aprende imputação e vocabulário apenas no treino. A validação recebe `transform`. Em `02...`, `medir` ajusta filtro/PCA e executa `fit_resample` somente no treino. Nas redes, o oversampling também ocorre depois da preparação e somente no conjunto de ajuste. Essas práticas respeitam a separação recomendada para transformações aprendidas. [Referência: prevenção de vazamento](https://scikit-learn.org/1.7/common_pitfalls.html#data-leakage).

### Limite histórico da quinta divisão

`DECISOES.md:57` e a célula `dados-nota` de `02...` informam que a quinta divisão já integrou médias de um piloto. A Etapa 2 avalia 13 combinações nela (`CONFIRMACAO`); depois ela aparece nas comparações SVM, RF, redes com escolha de épocas, redes com 120 épocas e PyTorch. Os notebooks da Etapa 3 reconhecem sua reutilização.

Há evidência documental de **reutilização da avaliação durante o desenvolvimento**, com risco de adaptação das decisões aos resultados. Isso impede apresentar a quinta divisão como teste intocado ou estimativa final independente. Não demonstra que `fit` recebeu suas linhas nas buscas atuais, nem permite medir o tamanho do eventual otimismo. Também não prova acesso ao teste oficial: os caminhos examinados leem apenas o treino.

A futura consolidação deve preservar a quinta como referência histórica e declarar seu papel. Uma validação aninhada por grupos pode separar seleção e avaliação no novo protocolo, mas não apaga o conhecimento adquirido com as análises anteriores. Não basta trocar a semente e chamar a nova divisão de teste nunca observado. O teste oficial permanece reservado.

### As três colunas sob investigação

| Coluna | Decisão existente e parecer |
| --- | --- |
| `score_ids_legado` | Excluída provisoriamente. Pode ser legítima se vier de detector anterior independente e estiver disponível na inferência; pode vazar se construída com rótulos ou informações futuras. Falta proveniência. |
| `tipo_alerta_soc` | Excluída provisoriamente. Confirmar se é alerta anterior à classificação ou produto de investigação posterior. Associação elevada não resolve essa questão. |
| `tempo_resposta_analista_min` | Excluída provisoriamente, assumindo tempo observado de atendimento posterior à conexão. Se for estimativa/histórico disponível previamente, reavaliar a premissa. O padrão de ausência também pode carregar informação posterior. |

As exclusões aparecem em `COLUNAS_EXCLUIDAS_EXPERIMENTOS` em todas as seis cópias; as colunas permanecem na EDA. **Não há prova de vazamento temporal dessas colunas disponível no repositório.** A exclusão conservadora é coerente com a incerteza documentada. Imputação, correlação baixa ou presença da coluna no teste não demonstrariam sua legitimidade.

## 4. Parecer sobre os classificadores

| Modelo | Implementação conferida | Parecer e limites |
| --- | --- | --- |
| Random Forest | `definicoes-floresta`: 100 árvores, Gini, `max_features="sqrt"`, `min_samples_leaf=2`, profundidade 12/livre, pesos ausentes/`balanced_subsample`, semente 42. Seleção por F1-macro médio, desempate por recall, antes da quinta. | Coerente e funcional no teste sintético das quatro opções. Profundidade livre não prova overfitting. Faltam artefatos reais e diagnóstico de generalização. |
| SVM linear | `definicoes-svm`: `LinearSVC(dual=False, max_iter=5000)`, C de 0,1/1/10 e alternativa ponderada em C=1. | Coerente para a matriz tabular; sete opções SVM executaram no teste sintético. LinearSVC usa formulação/solver e perda padrão diferentes do SVC RBF: não é comparação que altera apenas o kernel. |
| SVM RBF CPU | C de 1/10, `gamma="scale"`, limite de 200.000 iterações; uma opção ponderada. A busca prioriza convergência em todas as quatro rodadas. | Correta no código inspecionado e no teste funcional. Na quinta, convergência é informada, mas não impede publicar a métrica se falhar. Os limites e os tempos históricos precisam dos logs reais. |
| MLP pequena/profunda | `definicoes-redes`: (16,) e (64,32,16), ReLU, Adam, lote 512, taxa 0,001, `alpha=0.0001`, `RandomState(42)` mutável por modelo; 120 chamadas de `partial_fit`, sem early stopping. | Retropropagação e continuidade do otimizador são coerentes; não há reinicialização manual por época. Ambas completaram 120 épocas no teste sintético, com as assertions de `loss_curve_` e `t_`. Os resultados antigos de escolha de épocas não são gerados pelo código atual. |
| PyTorch | `definicoes-pytorch`: (128,64,32,16), ReLU, dropout 0,15 após as três primeiras camadas, saída de um logit, `BCEWithLogitsLoss`, Adam com taxa 0,001 e `weight_decay=0.0001`. | Sequência `zero_grad → forward → backward → step` correta por inspeção. `train()` ativa dropout; `eval()` e `inference_mode()` o desativam na previsão. Logit >= 0 corresponde a probabilidade >= 0,5. Não executado nesta auditoria. |
| SVM RBF CUDA | `definicoes-benchmark`: mesmos arrays salvos em NPZ, float64 nas entradas, gamma explícito `1/(n_features * X.var())`, C=10, tolerâncias 1e-3/1e-4 e `max_iter=-1`. | Desenho de benchmark coerente por inspeção; sincroniza GPU e separa transferências, ajuste e previsão. Somente o caminho CPU foi testado sinteticamente. Não há resultado cuML reproduzido ou desempenho CUDA comprovado. |

O uso de `partial_fit` corresponde a uma passagem pelo conjunto por chamada. Os números iguais de regularização das MLPs e do Adam PyTorch não tornam as penalizações equivalentes: a definição de `alpha` inclui normalização pelo número de amostras, além de haver dropout e implementações distintas. A comparação deve ser descrita como comparação de pipelines. [Referência: MLPClassifier 1.7](https://scikit-learn.org/1.7/modules/generated/sklearn.neural_network.MLPClassifier.html).

Nas redes, o oversampling 0,5 busca uma anomalia para dois normais no ajuste; SVM e RF não o usam. RF pode ponderar por amostra bootstrap; SVM testa pesos de classe. Isso não invalida a comparação de soluções completas, mas não permite atribuir diferenças exclusivamente à arquitetura. As duas MLPs são variações da mesma família, e a rede PyTorch também é uma MLP tabular, apesar do nome deep learning. RF, SVM e redes já fornecem três famílias distintas; a árvore da Etapa 2 é uma referência adicional.

Reprodutibilidade: sementes de divisão/modelos/amostragem estão definidas. PyTorch usa `torch.manual_seed`, gerador explícito, `torch.use_deterministic_algorithms(True)` e `CUBLAS_WORKSPACE_CONFIG`. Isso ajuda dentro de um ambiente, sem garantir igualdade entre versões, CPU e GPU. [Referência: PyTorch 2.10](https://docs.pytorch.org/docs/2.10/notes/randomness.html).

No benchmark CUDA, a CPU roda por padrão só a divisão 1 e a GPU as quatro primeiras. Os pares de tempo e concordância usam corretamente divisão e tolerância iguais. As médias gerais de uma e quatro divisões não devem ser comparadas como se estivessem pareadas. Uma execução por par também não estabelece aceleração estável. O cuML tem critério de estagnação além da tolerância, reconhecido pelo notebook; igualdade de parâmetros não garante previsões ou número de iterações iguais. [Referência: cuML SVC](https://docs.nvidia.com/cuml/latest/api/generated/cuml.svm.SVC/).

## 5. Métricas, resultados registrados e overfitting

### Cálculos e matrizes

O código usa `f1_score(..., average="macro")` ou `classification_report(...)["macro avg"]["f1-score"]`. Isso corresponde à média não ponderada do F1 de cada classe, não à acurácia, ao F1 ponderado ou ao F1 só de `anomaly`. Precisão e recall têm `anomaly` como classe positiva, diretamente ou por codificação `normal=0`, `anomaly=1`.

As matrizes usam `labels=["normal", "anomaly"]` ou `[0,1]` e são exibidas com linhas reais e colunas previstas. Assim, são `[[TN, FP], [FN, TP]]` para anomalia positiva. A ordem está consistente entre cálculo e apresentação. Na futura consolidação, convém fixar os dois rótulos também nas chamadas de `f1_score` e validar a presença das duas classes em cada divisão; hoje RF/SVM dependem das classes observadas nessa chamada.

Conferência realizada **somente com as contagens publicadas**, usando 13.095 normais e 788 anomalias da quinta divisão:

| Registro em `DECISOES.md` | Matriz reconstruída das contagens | F1-macro recalculado | Recall anomaly | Precisão anomaly |
| --- | --- | ---: | ---: | ---: |
| SVM RBF, linha 95 | `[[12926,169],[218,570]]` | 0,865906 | 72,3350% | 77,1313% |
| RF, linha 103 | `[[12889,206],[156,632]]` | 0,881760 | 80,2030% | 75,4177% |
| MLP pequena inicial, linha 116 | `[[12771,324],[155,633]]` | 0,853547 | 80,3299% | 66,1442% |
| PyTorch, linha 139 | `[[12760,335],[159,629]]` | 0,849523 | 79,8223% | 65,2490% |

Os arredondamentos correspondem ao documento. Fórmula utilizada: `F1_macro = (2TP/(2TP+FP+FN) + 2TN/(2TN+FP+FN))/2`. Isso confirma consistência aritmética, não a origem das previsões ou a qualidade medida numa nova execução.

### Comparação histórica, sem novos resultados

| Configuração documentada | Média das quatro divisões | Quinta divisão | Estado da evidência |
| --- | ---: | ---: | --- |
| RF profundidade livre, balanced_subsample | 0,8744 | 0,8818 | Código compatível; aritmética da quinta confere; execuções não reproduzidas. |
| SVM RBF C=10 sem pesos | 0,8506 | 0,8659 | Código compatível; aritmética da quinta confere. |
| Melhor linear, C=0,1 | 0,8013 | 0,8168 | Configuração existe; faltam previsões/contagens para conferência independente. |
| MLP pequena com escolha de épocas | 0,8474 | 0,8535 | Histórico de metodologia anterior; código de seleção de épocas não está no notebook atual. |
| MLP profunda com escolha de épocas | 0,8420 | 0,8519 | Mesma limitação histórica. |
| MLP pequena, 120 épocas | 0,8468 | 0,8512 | Configuração atual; números ainda dependem dos artefatos históricos. |
| MLP profunda, 120 épocas | 0,8441 | 0,8495 | Configuração atual; números ainda dependem dos artefatos históricos. |
| PyTorch, 120 épocas | 0,8514 | 0,8495 | Configuração atual; aritmética da quinta confere. |

As médias, desvios e afirmações de vitória em três/quatro divisões não podem ser reconstruídos a partir de um valor agregado. A comparação CPU de tolerâncias (0,8465, previsões idênticas e números de iterações em `DECISOES.md:97`) também exige logs/previsões para confirmação. Não existe evidência publicada de ganho CUDA nessa seção; o próprio documento diz que a medição está pendente.

A escolha **preliminar** do RF é fundamentada pelos números registrados: maior média das quatro divisões e melhor F1-macro na quinta; frente à SVM, as contagens publicadas indicam 62 anomalias adicionais detectadas e 37 falsos alarmes adicionais. Não é conclusão de superioridade estatística ou garantia de desempenho no teste oficial. A preparação é comparável, mas há diferenças de balanceamento, busca, regularização e orçamento de treino.

### Overfitting: conclusão possível

Não há evidência suficiente para afirmar ou descartar overfitting. RF de profundidade livre e redes maiores têm capacidade para memorizar, mas capacidade não é diagnóstico. Os notebooks das redes guardam perda de treino por época e F1 de validação apenas no final; não guardam curvas pareadas suficientes para localizar divergência entre ajuste e generalização. RF/SVM não produzem comparação sistemática de desempenho de treino e validação.

As pequenas quedas históricas após 120 épocas não comprovam overfitting: também houve mudança no protocolo de épocas, e faltam execuções/curvas pareadas. A MLP de apenas 16 iterações da Etapa 2 atingia o limite, conforme documentado; suas conclusões sobre escala e combinações são preliminares e não devem ser tratadas como otimização definitiva das redes de 120 épocas.

## 6. Achados priorizados

Gravidade crítica significa comprometimento direto demonstrado dos resultados centrais; alta impede uma conclusão independente ou sua rastreabilidade; média afeta robustez, diagnóstico ou reprodução; baixa afeta documentação ou caso periférico. **Nenhum achado crítico foi demonstrado.** Limitações e riscos são separados de falhas reproduzidas.

| ID | Gravidade | Evidência, impacto e encaminhamento |
| --- | --- | --- |
| A01 | Alta | **Quinta divisão reutilizada.** `DECISOES.md:57,107–139`; `02...`, `dados-nota` e `CONFIRMACAO`; protocolos de `03...`. Não é teste intocado. Separar seleção/avaliação no protocolo futuro e manter o resultado antigo explicitamente histórico. Não há demonstração de sobreposição no ajuste atual. |
| A02 | Alta | **Rastreabilidade incompleta das métricas e da versão antiga das redes.** Todos os notebooks têm zero saídas; JSON/CSV ficam em `tempfile.mkdtemp`; diretórios históricos mencionados não existem neste checkout. `DECISOES.md:107` descreve escolha interna de épocas, mas `definicoes-redes` contém apenas `EPOCAS=120`. O histórico Git disponível desse notebook só mostra sua introdução em `1c907e4`. Recuperar artefatos/receita antiga ou rotular números antigos como não reproduzíveis com a versão atual; não sobrescrevê-los com resultados sintéticos. |
| A03 | Média | **Manifestos e proveniência incompletos.** `requirements.txt` cobre só EDA; células `ambiente` fixam outro conjunto, incluindo NumPy 2.5.4/SciPy 1.18.1, que exigem Python >=3.12 nos metadados instalados. A instrução Python >=3.11 do README é da Etapa 1 e não deve ser estendida às demais. Etapa 2 registra SHA-256 do CSV; resultados da Etapa 3 não registram uniformemente hash dos dados, commit, índices das divisões e todas as versões. Explicitar ambientes por etapa e salvar um manifesto por execução. Os pins CPU testados são instaláveis e passaram em `pip check`. |
| A04 | Média | **Ambiente GPU não estritamente fixado.** `03_deep_learning...`, `ambiente-pytorch`: um torch já importável pode ser aceito sem igualdade com `versao_referencia="2.10.0"`. `03_svm_cuda...`, `rodar-cuda`: o subprocesso parte de `os.environ`, sem herdar o conjunto CPU verificado; fixa cuML, mas não registra/verifica todo o ambiente. Verificar compatibilidade e registrar versões efetivas antes de comparar. Não foi comprovada falha de instalação GPU nesta auditoria. |
| A05 | Média | **Diagnóstico de overfitting insuficiente.** `03_redes_neurais.../avaliar` e `03_deep_learning.../avaliar` guardam `perda_treino` por época e medem validação ao final. Acrescentar diagnóstico de treino/validação em divisões internas, sem transformar a quinta em critério de parada. |
| A06 | Média | **Coluna categórica totalmente ausente com moda quebra o transformador.** Nas seis células `definicoes-tecnicas`, `SimpleImputer(strategy="most_frequent")` não mantém colunas vazias, mas o laço de one-hot percorre `colunas = list(treino)` original. Reproduzido com `service` totalmente ausente no treino e `flag` preenchida: `ValueError` por array com zero atributos. Definir tratamento e preservar alinhamento de colunas. Não há evidência de que essa situação ocorreu na base oficial. |
| A07 | Média | **KNN com coluna numérica totalmente ausente devolve valores não finitos.** Em `transformar_numericas`, `StandardScaler().fit(treino)` mantém estatísticas indefinidas nessa coluna; `KNNImputer(keep_empty_features=True)` não resolve o posterior `inverse_transform`. Reproduzido: saída contém NaN e seria rejeitada pela verificação antes do modelo. Tratar explicitamente colunas vazias com estatísticas/política aprendidas só no treino. Impacto real ainda não demonstrado. |
| A08 | Média | **Duplicação e lacuna de testes das Etapas 2/3.** As seis cópias compartilham funções equivalentes hoje, mas `tests/test_preprocessamento.py` importa somente o módulo de diagnóstico. As falhas A06/A07 não são cobertas. Criar fonte comum ou geração controlada preservando o uso independente no Colab, com testes de isolamento, rótulos, agrupamento e casos de ausência. |
| A09 | Média | **Convergência não é condição de aceitação em todas as avaliações.** `03_svm.../main` filtra a escolha nas primeiras quatro divisões, mas o resultado da quinta apenas recebe `convergiu`. `03_svm_cuda.../avaliar` registra estado e segue calculando métricas. Exigir estado válido ou sinalização de resultado não elegível; não confundir execução terminada com convergência. Nenhum ajuste SVM do teste sintético falhou em convergir. |
| A10 | Baixa | **Link e descrição de PDF desatualizados.** `README.md:45` aponta para `output/pdf/relatorio_etapa1.pdf`, ausente; os arquivos presentes são `etapa1_resumo.pdf` e `etapa2_resumo.pdf`, com uma página cada. A descrição de capa/sumário/várias páginas não corresponde ao artefato presente. Atualizar documentação em tarefa posterior. |
| A11 | Baixa | **Redundância de protocolo sem observações comparáveis.** `preprocessamento.py:247–251`: `.le(1).all()` em pares vazios retorna verdadeiro e o relatório indica 100%/`redundancia_comprovada=True` para zero pares. Reproduzido com ambas as colunas ausentes. Exigir ao menos um par, como já ocorre nas relações numéricas. Não altera automaticamente os modelos, mas pode produzir diagnóstico indevido em outra base. |

Além desses achados, o cabeçalho inicial dos notebooks verifica só seis colunas e `preparar_base` pressupõe outras pelo nome e pelas contagens 45/41/40. Isso é adequado ao formato esperado, mas a próxima fase deve validar o esquema completo e produzir mensagens claras para colunas ausentes, adicionais ou tipos inesperados. Não se deve inferir o esquema real a partir da fixture sintética usada nesta auditoria.

## 7. Atendimento dos requisitos de pré-processamento

| Requisito solicitado | Implementação existente | Situação |
| --- | --- | --- |
| Dados redundantes e inconsistentes | Versões A/B/C; remoção da constante; padronização nominal; marcação de valores negativos/taxas inválidas; comparação de deduplicação e exclusão de conflitos somente no treino. | Atendido no código, com resultados históricos. `su_attempted=2` e extremos são preservados deliberadamente até esclarecimento; não assumir que todo extremo deve ser removido. A06/A07/A11 limitam robustez. |
| Pelo menos duas técnicas de balanceamento | `RandomOverSampler` e `RandomUnderSampler`, `sampling_strategy=0.5`, comparados à ausência de amostragem em `02.../medir`. | Atendido no código; ambos atuam só no treino. Comparação histórica usa principalmente árvore, com combinações adicionais na rede. O melhor tratamento depende do modelo. |
| Conversão simbólico-numérica | One-hot completo, agrupamento de `service` raro e `OrdinalEncoder` com desconhecido=-1; `protocolo_cod` nominal na versão A. | Atendido; one-hot não impõe ordem, categorias novas têm política explícita. Inteiros impõem ordem artificial, discutida no notebook. |
| Normalização/transformação numérica | Z-score, Min-Max, RobustScaler e ausência de escala; ajuste dentro da divisão. | Atendido; evidência de escolha da escala usa MLP com orçamento de 16 iterações. Confirmar adequação no protocolo de consolidação. |
| Redução de dimensionalidade ou seleção de instâncias | Remoção de redundâncias, deduplicação, `SelectKBest(f_classif)` e PCA de até 60 dimensões; filtro/PCA ajustados apenas no treino. | Técnicas implementadas e comparadas. Pipelines recomendados não usam PCA/filtro adicional, com justificativa em F1. Se a rubrica exigir uso final obrigatório, isso precisa de confirmação do enunciado original; não presumir essa exigência. |
| Pelo menos três algoritmos | Random Forest, SVM e redes; árvore individual adicional na Etapa 2. | Atendido em implementação e no registro histórico. Falta comparação consolidada rastreável; duas MLPs não devem ser contadas como famílias independentes apenas por profundidade. |

Pendências principais: confirmar proveniência dos três sinais excluídos; esclarecer `su_attempted=2` sem recodificação arbitrária; recuperar evidências reais; definir avaliação que reconheça o uso histórico da quinta; resolver robustez/dependências e integrar a consolidação numa próxima tarefa autorizada.

## 8. Testes, ambiente e limitações

Ambiente: Windows, Python **3.12.10**, ambiente virtual temporário fora do repositório em `%TEMP%\nsl-kdd-stage3-audit-20261122`. Não houve alteração do Python global nem dos manifests do projeto. GPU detectada: RTX 3050 6GB Laptop, driver 581.57; sua presença não significa que PyTorch/cuML tenham sido exercitados. A detecção do notebook cuML retorna CPU em Windows nativo por verificar plataforma Linux.

Dependências CPU instaladas para os testes: pandas 3.0.1, NumPy 2.5.4, SciPy 1.18.1, scikit-learn 1.7.2, imbalanced-learn 0.14.1, joblib 1.6.0, threadpoolctl 3.8.0 e sklearn-compat 0.1.6. Os testes da EDA foram executados nesse ambiente dos experimentos, **não** no conjunto exato de `requirements.txt` (NumPy 2.4.2 e bibliotecas gráficas).

| Verificação executada | Resultado e alcance |
| --- | --- |
| `py -3.12 -m unittest discover -s tests -v` no ambiente inicial | Falhou na importação por ausência de pandas; não foi falha de asserção do projeto. |
| Mesma suíte após concluir instalação no ambiente isolado | **10 testes passaram**, última execução em 0,058 s. Casos sintéticos de diagnóstico, não validação de desempenho. |
| `python -m pip check` no ambiente isolado | `No broken requirements found.` Imports e versões CPU confirmados após a instalação. |
| JSON, IDs e compilação de células | Oito notebooks lidos, IDs únicos por notebook, 70 células de código compiladas. Magias IPython transformadas; corpo de `%%writefile` compilado como Python. Sem executar notebooks inteiros. Três arquivos Python também compilaram em memória. |
| Validação formal por nbformat no ambiente global | Tentativa interrompida no import de `referencing`: `TypeError: 'default' is an invalid keyword argument for typevar()`. Problema do ambiente global; não é evidência de JSON inválido. A verificação acima não equivale à validação completa pelo schema nbformat. |
| Consistência das cópias | Comparação AST das funções compartilhadas nas seis cópias; módulo da EDA e nove células de análise coincidem com as fontes. Gerador independente executado em memória, saídas vazias confirmadas. |
| Fixture sintética de 240 linhas/120 grupos | Preparação A/B/C, preservação da entrada, normalização, máscaras de domínio, agrupamento ignorando rótulos, cobertura das validações, disjunção de grupos/índices, exclusão da reserva e repetibilidade da divisão passaram. Deduplicação preservou representantes de rótulos conflitantes; remoção de conflitos removeu o grupo apenas do ajuste. |
| Transformações sintéticas | 16 combinações numéricas e seis categóricas passaram nos casos comuns. Mediana/escala conferidas contra cálculo manual com valor extremo somente na validação. Ausência `pd.NA` e categoria desconhecida também verificadas. Casos totalmente ausentes revelaram A06/A07. |
| Modelos CPU sobre fixture sintética | Quatro RFs e sete SVMs executaram; convergência SVM confirmada. Duas MLPs completaram 120 épocas e passaram nas verificações de épocas/amostras e dimensões das matrizes. Benchmark CPU executou ambas as tolerâncias. **Métricas dessa fixture não são resultados NSL-KDD e não entram nas tabelas históricas.** |
| Redução/amostragem sintéticas | Nove combinações: nenhuma/filtro/PCA × nenhum/over/under. Matriz de treino 120×75: reduções produziram 60 colunas; amostragens produziram respectivamente 120/150/60 linhas. Validação e entradas originais preservadas. |
| Aritmética documental | Matrizes reconstruídas de SVM, RF, MLP pequena inicial e PyTorch reproduzem os arredondamentos de `DECISOES.md`. Não verifica as previsões reais. |
| Caso sem pares de protocolo | Confirmou A11: zero pares e redundância declarada verdadeira. |

Os módulos dos notebooks foram carregados diretamente de suas células em memória, sem editar fontes. Os experimentos sintéticos verificam funcionamento e invariantes; não avaliam o conjunto oficial, não demonstram ausência universal de vazamento e não comprovam comparabilidade histórica dos arquivos de entrada.

### Reexecutar os testes existentes

Na raiz do repositório, em PowerShell, reutilizando o ambiente da auditoria:

```powershell
$auditPython = Join-Path $env:TEMP 'nsl-kdd-stage3-audit-20261122\Scripts\python.exe'
& $auditPython -m pip check
& $auditPython -m unittest discover -s tests -v
```

Para recriar o ambiente, caso ele tenha sido removido:

```powershell
$auditEnv = Join-Path $env:TEMP 'nsl-kdd-stage3-audit-20261122'
py -3.12 -m venv $auditEnv
$auditPython = Join-Path $auditEnv 'Scripts\python.exe'
& $auditPython -m pip install pandas==3.0.1 numpy==2.5.4 scipy==1.18.1 scikit-learn==1.7.2 imbalanced-learn==0.14.1 joblib==1.6.0 threadpoolctl==3.8.0 sklearn-compat==0.1.6
& $auditPython -m unittest discover -s tests -v
```

Para conferir A06/A07 sem gerar arquivos, carregar os módulos em memória no mesmo Python e usar os dois casos abaixo, separadamente (o primeiro lança exceção):

```python
import json, sys, types
from pathlib import Path
import numpy as np
import pandas as pd

nb = json.loads(Path("02_preprocessamento_colab.ipynb").read_text(encoding="utf-8"))
for nome, cell_id in [("dados_etapa2", "definicoes-dados"),
                      ("tecnicas_etapa2", "definicoes-tecnicas")]:
    cell = next(c for c in nb["cells"] if c.get("id") == cell_id)
    modulo = types.ModuleType(nome)
    sys.modules[nome] = modulo
    exec("".join(cell["source"]).split("\n", 1)[1], modulo.__dict__)
t = sys.modules["tecnicas_etapa2"]

# A06: ValueError na configuração moda + one-hot.
t.transformar_categoricas(
    pd.DataFrame({"service": [np.nan, np.nan], "flag": ["sf", "s0"]}),
    pd.DataFrame({"service": ["novo"], "flag": ["sf"]}), "moda", "onehot")

# A07: executar separadamente após o carregamento; a/b contêm NaN.
a, b = t.transformar_numericas(
    pd.DataFrame({"a": [np.nan, np.nan], "b": [1., 2.]}),
    pd.DataFrame({"a": [3.], "b": [1.]}), "knn", "zscore")
print(np.isfinite(a).all(), np.isfinite(b).all())
```

## 9. Recomendações e plano da próxima fase

Este é um plano, **não uma implementação realizada ou autorização para avançar**. Preservar notebooks e resultados históricos durante a consolidação.

1. **Prioridade 1 — congelar evidências e protocolo.** Obter o treino oficial por canal autorizado, conferir esquema/hash e recuperar JSON/CSV/logs existentes antes de gastar recursos repetindo ajustes. Marcar claramente experimento antigo, ambiente, semente, política de épocas e papel da quinta divisão. Escolher um protocolo agrupado com seleção interna e avaliação externa, reconhecendo o histórico; limitar previamente orçamento e configurações. Não usar o teste oficial para decidir.
2. **Prioridade 1 — tornar o preparo testável e robusto.** Centralizar ou gerar as funções comuns mantendo notebooks independentes no Colab. Tratar colunas inteiramente ausentes, preservar correspondência de nomes/dimensões e validar classes/esquema. Acrescentar testes de isolamento do `fit`, disjunção dos grupos, categorias novas, deduplicação só no treino e preservação da validação. Resolver A11 separadamente no diagnóstico.
3. **Prioridade 1 — garantir reprodução do ambiente.** Criar manifesto CPU específico da Etapa 3 e separar dependências opcionais PyTorch/cuML. Exigir Python compatível, verificar versões efetivas e registrar hardware. Não modificar o manifesto da EDA indiscriminadamente. Verificar GPU em ambiente suportado antes de agendar experimento; CPU continua caminho disponível.
4. **Prioridade 2 — construir executor de avaliação comum.** Reutilizar RF, SVM e MLPs existentes. Fixar os mesmos índices/grupos e a ordem dos rótulos; ajustar cada pipeline somente no treino correspondente. Registrar política de balanceamento, hiperparâmetros, convergência, contagens, F1-macro, precisão/recall por classe, matriz e tempos. Persistir índices/previsões em local privado quando puderem revelar dados; versionar apenas resumos e metadados adequados.
5. **Prioridade 2 — consolidar comparação rastreável.** Produzir tabela por divisão e pipeline, média/desvio e matrizes com suportes explícitos. Não misturar redes com escolha de épocas e redes de 120 épocas; não agregar quinta e quatro primeiras como avaliações equivalentes. Comparar pares nas mesmas divisões, declarar diferenças de preparação e não transformar desvio entre folds em prova automática de significância.
6. **Prioridade 2 — investigar generalização.** Adicionar métricas de ajuste e validação e curvas em divisões internas para redes; manter seleção de épocas dentro do treino externo se for retomada. Reavaliar RF como candidato a partir desse protocolo, sem substituir automaticamente o escolhido por uma pequena oscilação de F1.
7. **Prioridade 3 — benchmark GPU e documentação.** Medir CPU/CUDA nos mesmos pares, com repetição controlada se necessário, estado de convergência e ambiente registrados. Separar benchmark de tempo da seleção de algoritmo. Atualizar links do README e associar cada tabela histórica a seus artefatos.

Arquivos candidatos para uma tarefa futura: módulos de experimentos comuns em estrutura a definir após revisar o padrão existente; testes específicos em `tests/`; manifesto separado de dependências da Etapa 3; ajustes mínimos nas células dos notebooks para usar/gerar a fonte comum; `README.md` e `DECISOES.md` para documentação. Não há aplicação web, rotas, banco ou necessidade de SQL. Nomes e divisão final dos arquivos devem ser apresentados no plano da implementação, com commits pequenos por arquivo.

Critérios de conclusão da próxima fase: mesmas divisões verificáveis entre pipelines; ausência de ajuste de transformadores na validação; métricas recalculáveis das previsões; dados/versões/configurações rastreáveis; resultados históricos separados dos novos; execução funcional dos modelos selecionados e cobertura dos casos de borda. A Etapa 4, notebook final, CSV de submissão e relatório acadêmico final permanecem fora do escopo.

## 10. Encerramento desta fase

A estrutura central dos classificadores e o isolamento das transformações estão coerentes. RF, SVM e MLPs scikit-learn tiveram funcionamento verificado em dados sintéticos; PyTorch e cuML receberam parecer estático, sem execução. O RF permanece candidato preliminar com base no registro histórico. Os principais obstáculos são independência da avaliação, rastreabilidade dos resultados e robustez nos casos descritos, não uma demonstração de treinamento direto sobre a validação.

Foram preservados código, notebooks, testes, PDFs, dados e funcionalidades existentes. Não foram criados modelos, comparação consolidada executável, notebook final, submissão ou SQL. Commit documental previsto nesta branch: `docs: audit stage 3 model implementations`. A continuidade depende de nova instrução para a Fase 2.
