# Etapa 3 — Execução e comparação reproduzível

## Dados e escopo

Base oficial confirmada pelo usuário: `C:\Users\Costa\Downloads\treino (1).csv`.
Esse arquivo, `Downloads\treino.csv` e a cópia local em `data/raw/treino.csv` têm o mesmo SHA-256:

`9713c75938ef1e9e513975f256d65a80c4b05df533d35336c67387020bf76b7f`.

São 69.833 registros, 49 atributos e `class`: 65.754 `normal` e 4.079 `anomaly`.
O esquema completo e os tipos foram conferidos, e as representações A/B/C produziram
45/41/40 atributos antes da codificação. A representação C forma 56.766 grupos.
O hash identifica a cópia confirmada; não existe no projeto um hash autenticado publicado
pela professora para comprovar a história anterior desse arquivo. A execução verifica que
o conteúdo permanece inalterado. Nenhum CSV oficial é versionado.

O trabalho usa somente o treino. O teste sem rótulos, a submissão, explicabilidade,
relatório acadêmico final, apresentação e limpeza da Etapa 4 estão fora desta fase.

## Protocolo e orçamento

Cinco divisões externas e duas internas por treino externo, com `StratifiedGroupKFold`
e semente 42. As cinco validações externas cobrem a base completa, uma vez por comparação.
Entradas iguais em C, inclusive com classes conflitantes, permanecem no mesmo grupo.
As mesmas divisões externas são usadas em todos os candidatos, mesmo quando se mantém
a representação A. Cada ajuste interno usa aproximadamente 40% da base; cada ajuste
externo usa aproximadamente 80%, sujeitos ao tamanho indivisível dos grupos.

Imputação, codificação, normalização, PCA, seleção de linhas e balanceamento usam somente
o treino correspondente. Deduplicação considera entradas e rótulo dentro do ajuste;
não exclui linhas da validação. O CSV original e as três colunas sob investigação de
proveniência são preservados no disco; as exclusões aprovadas permanecem nos modelos.

O plano foi fixado antes de observar as métricas: 15 configurações, 150 ajustes internos
e até 60 externos. Ajustes externos idênticos são reutilizados entre comparações,
com registro de todas as comparações atendidas. Dentro de cada família, a configuração
é escolhida pelos folds internos, por F1-macro médio, recall de `anomaly` e ordem declarada.
A linha `selecao_global_aninhada` aplica a mesma regra ao conjunto dos 15 candidatos.
SVM sem convergência não pode ser selecionada. As redes completam exatamente 120 épocas,
sem escolher época pelos resultados externos.

Máquina local: Windows, Intel Core i5-13420H, 8 núcleos/12 threads, cerca de 16 GB de RAM.
Havia aproximadamente 3,4 GB de RAM livre no planejamento. Os ajustes são sequenciais,
com uma thread numérica e RF `n_jobs=1`. Não se reduz a amostra oficial para acelerar.
Python 3.12.10 e dependências de `requirements-stage3-report.txt`, que inclui os pins CPU
da Fase 2. PyTorch/cuML não estavam instalados; sua instalação e treinamento foram
dispensados para priorizar a comparação completa RF/SVM/MLP no orçamento definido.

## Configurações efetivamente previstas

| Comparação | Configurações |
| --- | --- |
| Random Forest | 100 árvores, Gini, sqrt, folha mínima 2; profundidade 12/livre × sem pesos/`balanced_subsample`. |
| SVM | LinearSVC C=0,1, dual=False, limite 5.000; SVC RBF C=10, gamma=scale, tolerância 0,001, limite 200.000, cache 256 MB. |
| MLP pequena | (16,), ReLU, Adam, batch 512, alpha=0,0001, taxa 0,001, 120 épocas, oversampling 0,5. |
| MLP profunda | (64,32,16), demais opções iguais às da pequena. |
| RF de referência | Profundidade livre, sem pesos, C, mediana, ausência nominal explícita, one-hot, Z-score, sem duplicatas, sem amostragem e sem PCA. |
| Variações de preparo na RF | Um fator por vez: oversampling 0,5; undersampling 0,5; ordinal; sem escala; todas as linhas; representação A; PCA de até 60 componentes. |

As regras determinísticas de inconsistências continuam em todos os tratamentos.
A/C compara remoção de redundâncias; todas/sem duplicatas compara seleção de instâncias;
PCA avalia redução adicional ajustada no treino. Não se trata de busca exaustiva.
A referência sem balanceamento é mantida. Não se usa o número histórico de épocas escolhidas
como se fosse equivalente às redes atuais de 120 épocas.
O orçamento de busca não é idêntico entre famílias: as ablações de pré-processamento foram
concentradas na RF para conter o custo e isolar fatores. A escolha identifica o melhor
pipeline entre os candidatos avaliados, não o melhor ajuste possível de cada algoritmo.

## Evidências e reprodução

`stage3_evaluation.py` registra manifesto, versões, sementes, hashes dos fontes e dos dados,
índices, previsões, rótulos reais, elegibilidade, perdas MLP, F1 de ajuste e métricas de validação.
O F1 de ajuste é medido nas linhas selecionadas/deduplicadas antes da amostragem, para não
inflar o diagnóstico pela repetição artificial das anomalias. O gap de F1 é esse valor menos
o F1 da validação; é um diagnóstico de generalização, não uma prova isolada de sobreajuste.
Cada ajuste é persistido e sincronizado em disco. A retomada verifica a identidade do
experimento, preserva fragmentos incompletos em `.partial` e não repete ajustes concluídos.
Não execute processos concorrentes no mesmo destino.

`stage3_comparison.py` recalcula as métricas a partir dos registros, confere os índices
contra o manifesto e exige cobertura integral e elegibilidade para o ranking. Exporta
tabelas por divisão, médias/desvios, resultados internos por configuração e cinco gráficos.
Precisão/recall macro e métricas por classe usam sempre a ordem `normal`, `anomaly`.
O desvio entre folds é amostral (`ddof=1`), não um intervalo de confiança.
Matrizes somadas contêm previsões externas de todas as linhas; seu F1 agregado pode diferir
da média dos F1 dos folds. As médias internas usam parcelas sobrepostas e não são réplicas
independentes. Tempos são diagnósticos da mesma máquina, não benchmark com repetições controladas.

```powershell
python -m pip install -r requirements-stage3-report.txt
python stage3_comparison.py
python -m unittest discover -s tests -v
python preparar_notebooks_etapa3.py --check
```

Os registros completos ficam em `temp/stage3_runs/comparacao_cpu/`, fora do Git.
O notebook `03_comparacao_modelos_colab.ipynb` inclui todos os módulos e instala as
dependências necessárias em diretório separado quando preciso. Use upload no Colab e
execute tudo; não é necessário executar os notebooks individuais. Para retomar depois
de perder a sessão, use `NSLKDD_DESTINO` em uma pasta do Drive já montada.

O notebook versionado não contém saídas por registro. As cópias executadas de verificação
ficam em `temp/`. Os notebooks individuais e seus registros históricos são preservados.

## Resultado reproduzido na base oficial

Os 210 ajustes foram concluídos (150 internos e 60 externos), e a verificação posterior
confrontou 5.027.976 rótulos registrados com o CSV, além de índices, grupos, matrizes,
F1 e escolhas internas. As cinco divisões externas cobrem os 69.833 registros em cada
comparação. Todos os ajustes foram elegíveis; nenhum SVM foi aceito sem convergência.
Os números abaixo são médias e desvios amostrais entre cinco folds, sem arredondamento
no arquivo exportado. Acurácia, precisão/recall macro e por classe, matrizes, tempos,
configurações e resultados por divisão estão em `output/stage3/`.

| Pipeline | F1-macro médio ± desvio | Acurácia | Precisão anomaly | Recall anomaly | FP / FN somados |
| --- | ---: | ---: | ---: | ---: | ---: |
| RF ponderada escolhida na família | 0,878042 ± 0,006809 | 0,972494 | 0,750286 | 0,792497 | 1.075 / 846 |
| SVM RBF C=10 escolhida na família | 0,854939 ± 0,006502 | 0,969389 | 0,760448 | 0,694851 | 892 / 1.246 |
| MLP pequena, 120 épocas | 0,847189 ± 0,007514 | 0,963328 | 0,655863 | 0,784122 | 1.680 / 881 |
| MLP profunda, 120 épocas | 0,844761 ± 0,008903 | 0,962956 | 0,655827 | 0,773657 | 1.663 / 924 |
| **RF livre + oversampling 0,5** | **0,878817 ± 0,005054** | **0,972637** | **0,751176** | **0,794564** | **1.073 / 838** |

O F1-macro da RF com oversampling por fold foi 0,875897; 0,877639; 0,885645;
0,872875; 0,882032. Ela teve o maior F1-macro médio entre os pipelines avaliados,
o menor desvio entre as duas melhores RFs e oito falsos negativos e dois falsos
positivos a menos na soma das cinco validações. O custo médio de ajuste foi 8,01 s,
ante 5,21 s da RF ponderada, neste computador e sob este protocolo. O gap médio de
F1 treino–validação foi 0,006132, sem grande diferença sistemática entre as parcelas.

A mesma RF com oversampling liderou o F1-macro médio da seleção interna (0,875406),
à frente da RF ponderada (0,873485). Portanto a seleção global aninhada escolheu
`rf_over` nas cinco divisões externas. **Recomendação da Etapa 3: `rf_over`**, com
100 árvores, profundidade livre, folha mínima 2, `max_features=sqrt`, Gini, sem
ponderação de classe, representação C, mediana, categoria explícita para ausências,
one-hot, Z-score, remoção de duplicatas somente no ajuste e oversampling 0,5.

Na RF de referência sem pesos/amostragem, F1-macro foi 0,863520. As variações
isoladas tiveram: undersampling 0,868041; ordinal 0,867408; sem escala 0,863396;
todas as linhas 0,866722; redundâncias A 0,867617; PCA 0,859583. Essas ablações
mostram o efeito no modelo de referência, sem afirmar interação ou ótimo global.
PCA custou em média 79,68 s por ajuste externo, sem ganho nessa comparação.

Cinco gráficos acompanham as tabelas: F1 por algoritmo, precisão/recall, matrizes
de confusão, variação entre divisões e efeito do pré-processamento. As imagens foram
inspecionadas visualmente. O notebook consolidado inclui uma tabela agregada desta
execução e refaz os cálculos quando executado. As tabelas versionadas contêm somente
agregados; os registros por linha continuam em `temp/`, fora do Git.

Verificações executadas: `python -m unittest discover -s tests -v` (35 testes),
`python preparar_notebooks_etapa3.py --check`, `pip check`, validação do esquema
nbformat e execução sequencial das 16 células do notebook consolidado em kernel
limpo com o treino oficial. Essa execução reutilizou o checkpoint compatível,
recalculou a conferência e as figuras e concluiu sem erro de célula; a cópia
executada está em `temp/stage3_runs/notebook_executado.ipynb`, fora do Git.
Não houve execução em uma sessão Colab remota: o fluxo local verifica a sequência,
e as dependências do runtime Colab podem exigir instalação ao abrir o notebook.

O valor histórico de RF ≈ 0,8744 em `DECISOES.md` veio de outro protocolo e outra
composição de ajuste/validação. Ele confirma a RF como candidata preliminar, mas não
é somado nem subtraído como se fosse uma réplica da nova execução.

## Limites da escolha

A comparação externa é usada para recomendar o pipeline. Portanto a pontuação do vencedor
não é uma estimativa final totalmente independente: existe seleção entre alternativas.
A linha global aninhada avalia separadamente o procedimento de seleção interna, mas também
não apaga o conhecimento prévio da base durante o desenvolvimento. A quinta divisão histórica
já foi reutilizada. Números históricos não são combinados com a nova validação.

Continuam abertas a proveniência de `score_ids_legado`, `tipo_alerta_soc` e
`tempo_resposta_analista_min`, e a interpretação de `su_attempted=2`. Nenhuma dessas
questões foi resolvida por suposição. Os resultados se aplicam à base e ao orçamento
observados; não demonstram generalização para tráfego contemporâneo.
