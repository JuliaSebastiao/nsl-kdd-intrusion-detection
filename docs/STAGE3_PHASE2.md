# Etapa 3 — infraestrutura após a Fase 2

Este documento descreve o código preparado para a Fase 3. Não contém resultados novos da base oficial. O relatório `STAGE3_AUDIT.md` permanece como registro histórico da Fase 1.

## Ambiente

Use Python 3.12. As dependências necessárias ao fluxo principal em CPU estão fixadas em `requirements-stage3-cpu.txt`; `requirements.txt` continua atendendo a análise exploratória da Etapa 1. A suíte de testes foi executada com esse conjunto em Python 3.12.10 no Windows.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-stage3-cpu.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe preparar_notebooks_etapa3.py --check
```

PyTorch é **opcional**: a configuração `pytorch_profunda` importa `torch` somente ao treinar. Para reproduzir o notebook existente, ele exige PyTorch 2.10.0 (com variante CPU ou CUDA compatível) e registra a versão efetiva. Instale o wheel correspondente ao sistema seguindo o [seletor oficial do PyTorch](https://pytorch.org/get-started/locally/) em um ambiente separado. O fluxo principal RF/SVM/MLP scikit-learn não precisa dele.

cuML é **opcional e restrito ao benchmark CUDA** de `03_svm_cuda_colab.ipynb`, não à infraestrutura principal. O notebook prevê Linux, GPU NVIDIA e `cuml-cu12==26.6.0` ou a família CUDA detectada, instala em diretório temporário e verifica a versão efetiva do cuML e das dependências CPU. Consulte os [pré-requisitos do RAPIDS](https://docs.nvidia.com/datascience/install/) antes de executar. Windows nativo executa somente o trecho CPU desse notebook. Uma GPU detectada não implica que os pacotes estejam disponíveis ou que o benchmark foi validado.

## Fontes e notebooks independentes

`dados_etapa2.py` contém a leitura e preparação determinística A/B/C, os grupos e as divisões antigas. `tecnicas_etapa2.py` contém imputação, codificação e escala ajustadas por divisão. Ambos são fontes de manutenção das células `definicoes-dados` e `definicoes-tecnicas` nos seis notebooks das Etapas 2/3. `preparar_notebooks_etapa3.py` incorpora essas fontes nas células `%%writefile` e preserva a função `medir` específica do notebook da Etapa 2. As cópias geradas continuam executáveis no Colab sem envio de módulos locais.

Ao alterar uma fonte, sincronize um notebook por vez e confira seu diff antes do commit individual:

```powershell
py -3.12 preparar_notebooks_etapa3.py --notebook 03_random_forest_colab.ipynb
py -3.12 preparar_notebooks_etapa3.py --check
```

Para colunas categóricas totalmente ausentes no ajuste, a imputação por moda usa a categoria explícita `__ausente__` apenas nessa coluna; todas as colunas mantêm suas posições. Categorias novas seguem o `handle_unknown` do codificador: zeros no one-hot comum, grupo infrequente quando existir no one-hot com raros, código -1 no ordinal. Para KNN numérico, colunas totalmente ausentes no ajuste recebem zero em ambas as parcelas antes da escala e da imputação; valores observados só na validação nessas colunas não definem estatísticas. O valor zero é uma política técnica para informação inexistente no treino, não uma estatística extraída da validação. Os demais casos mantêm o algoritmo anterior.

## Protocolo de avaliação preparado

`stage3_evaluation.py` fornece:

- `planejar_divisoes`: `StratifiedGroupKFold` externo e, dentro de **cada** treino externo, um novo `StratifiedGroupKFold` interno. Os grupos são entradas iguais na versão C sem o rótulo. A escolha usa apenas F1-macro das validações internas e exige elegibilidade em todas elas; em empate usa recall de `anomaly` e depois a ordem declarada. O modelo escolhido é então ajustado no treino externo e avaliado na validação externa.
- `Configuracao`/`CONFIGURACOES_REFERENCIA`: parâmetros dos modelos existentes e suas políticas de preparo/balanceamento. As MLPs usam o esquema de 120 chamadas `partial_fit`; o adapter PyTorch preserva arquitetura (128, 64, 32, 16), dropout e treino de 120 épocas. Não foram criadas arquiteturas novas.
- `avaliar_divisao`: transformação e, quando configurado, oversampling somente no ajuste; F1-macro, acurácia, precisão/recall/F1 e suporte por classe, matriz na ordem fixa `normal`, `anomaly`, previsões, tempos e F1 do ajuste original para diagnóstico futuro de overfitting.
- `status_convergencia`: SVMs sem status suficiente ou sem convergência ficam inelegíveis para seleção. Suas métricas ainda são registradas para diagnóstico. Nos notebooks SVM, o estado também aparece nas tabelas; pares CPU/CUDA só são comparados quando ambos são elegíveis.
- `executar_avaliacao`: grava `manifesto.json` e acrescenta cada rodada concluída a `resultados.jsonl` em um diretório novo. O manifesto inclui tipo e SHA-256 dos dados, commit, Python, plataforma, versões efetivas dos pacotes, sementes de divisões/modelos/amostragem, configurações, índices de todas as divisões e hash do protocolo. As linhas registram previsões e índices da validação, além das métricas. `resultados_compativeis` exige mesmo tipo de dados, hash, protocolo e índices externos para parear resultados.

O executor aceita `official_training` apenas com caminho chamado `treino.csv` e alvo válido. Fixtures devem usar `synthetic_test` com um DataFrame; o manifesto distingue esses dois tipos. Guarde as saídas em `temp/stage3_runs/`, que está fora do Git. Não publique o CSV oficial, índices ou previsões de linhas oficiais sem revisar a política de dados da equipe.

A quinta divisão foi consultada em experimentos anteriores e continua sendo **referência histórica**, nunca um teste intocado. O novo plano aninhado separa seleção interna da medição externa *na execução futura*, mas não torna os folds externos independentes de todo o processo de desenvolvimento já ocorrido. A mudança de semente não apaga conhecimento prévio. O teste oficial sem rótulos não participa deste protocolo.

Exemplo de uso **somente quando a Fase 3 for autorizada e o treino oficial estiver disponível**:

```python
from stage3_evaluation import CONFIGURACOES_REFERENCIA, executar_avaliacao

manifesto, resultados = executar_avaliacao(
    "data/raw/treino.csv",
    CONFIGURACOES_REFERENCIA[:5],  # CPU; PyTorch opcional fica fora desta lista
    "temp/stage3_runs/primeira_execucao_cpu",
    tipo_dados="official_training",
)
```

Este exemplo mostra a interface; **não foi executado nesta fase**. Ele é potencialmente custoso porque a seleção interna treina várias vezes em cada divisão externa. Para a Fase 3, definir o orçamento e confirmar os arquivos oficiais antes de executar. O resultado da rede antiga com escolha de épocas permanece em `DECISOES.md` como histórico; ele não é misturado automaticamente com as redes atuais de 120 épocas.

## Estado e pendências

Os testes atuais usam pequenas fixtures sintéticas e verificam preparo, grupos, ausência de ajuste de estatísticas na validação, balanceamento, métricas, ordem da matriz, repetibilidade, inelegibilidade por não convergência, persistência e sincronização das cópias. Não demonstram desempenho real do NSL-KDD.

Continuam pendentes a confirmação da proveniência de `score_ids_legado`, `tipo_alerta_soc` e `tempo_resposta_analista_min`, o esclarecimento de `su_attempted=2`, a recuperação dos artefatos de execuções históricas e a medição CUDA em ambiente suportado. Na Fase 3 será necessário escolher o orçamento, executar os classificadores com o treino oficial, revisar os registros por divisão e então consolidar os resultados. Nenhuma rota, banco de dados ou SQL foi introduzido.
