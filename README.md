# NSL-KDD: Detecção de Intrusão em Redes

Projeto acadêmico da competição de IA. A primeira entrega é um relatório em PDF com o diagnóstico da base oficial do Canvas e o link para o notebook no GitHub, conforme o formato confirmado pela equipe.

## Base oficial

Coloque os arquivos oficiais do Canvas em `data/raw/`: `treino.csv`, `teste_sem_rotulo.csv` e `modelo_de_submissao.csv`. A pasta e esses nomes de arquivo estão no `.gitignore`; os dados não devem ser versionados. O notebook da Etapa 1 lê apenas `data/raw/treino.csv`, usa `class` como alvo e não depende de Kaggle, internet, `nslkdd.zip`, `KDDTrain+.txt`, `attack_type` ou `difficulty`.

## Executar no Google Colab

[![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JuliaSebastiao/nsl-kdd-intrusion-detection/blob/feat/preprocessamento-etapa2/01_analise_exploratoria_colab.ipynb)

1. Abra a cópia `01_analise_exploratoria_colab.ipynb` pelo botão acima.
2. Escolha **Ambiente de execução > Executar tudo**.
3. Se solicitado, selecione apenas o `treino.csv` oficial. As demais células continuam em sequência.

Essa cópia contém as funções de diagnóstico. Não é necessário clonar o repositório, enviar `preprocessamento.py` ou editar caminhos. O treino já enviado para a pasta inicial da sessão ou para `data/raw/` é reutilizado; caso não exista, o notebook abre o seletor de arquivo. Os CSVs continuam fora do Git e, em uma nova máquina de execução do Colab, precisam ser fornecidos novamente.

Para evitar reinstalações do ambiente durante a execução, a cópia usa as bibliotecas disponíveis e instala apenas as ausentes. Ela mostra as versões utilizadas, que podem diferir das versões fixadas em `requirements.txt` para a validação local. O funcionamento e a duração da máquina do Colab estão descritos no [FAQ oficial](https://research.google.com/colaboratory/faq.html).

O notebook modular `01_analise_exploratoria.ipynb` e `preprocessamento.py` são as fontes de manutenção. Para atualizar a cópia versionada do Colab após editar essas fontes:

```powershell
python preparar_entrega_etapa1.py --colab
```

O comando gera o arquivo sem saídas da base. A análise permanece a mesma; somente a preparação do ambiente e o carregamento do CSV são adaptados.

A cópia foi executada em kernels locais isolados com as bibliotecas de referência e com pandas 2.2.3, NumPy 2.4.6, Matplotlib 3.10.0 e Seaborn 0.13.2, versões da [lista publicada pelo Colab](https://github.com/googlecolab/backend-info/blob/main/pip-freeze.txt) consultada em 01/11/2026. As duas execuções produziram o mesmo diagnóstico e nove gráficos. O seletor do CSV foi simulado na validação local; a sessão em nuvem não foi acessada.

## Executar localmente

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt ipykernel
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Abra `01_analise_exploratoria.ipynb` a partir da raiz, selecione o Python da `.venv` no ambiente de notebooks e execute as células na ordem. Quem já usa outro ambiente de notebooks pode instalar `requirements.txt` nesse ambiente.

O notebook do Git é mantido sem saídas com registros da base. As cópias executadas ficam em `temp/`, fora do versionamento. O relatório em PDF é versionado em `output/pdf/`.

## Entrega da Etapa 1: relatório em PDF

O arquivo para entregar é [o relatório da Etapa 1](output/pdf/relatorio_etapa1.pdf). Ele contém os principais achados, limitações e links para este repositório e para [o notebook da análise](01_analise_exploratoria.ipynb). O notebook deve ser executado com `preprocessamento.py` na raiz do projeto, seguindo as instruções acima.

A apresentação acadêmica segue o [guia de normalização da PUC Minas (2025)](https://www.pucminas.br/biblioteca/DocumentoBiblioteca/ABNT-GUIA-COMPLETO-Elaborar-formatar-trabalho-cientifico.pdf), adaptado ao relatório da disciplina. A identificação informa Ciência da Computação, Belo Horizonte e os cinco integrantes do Trabalho Prático 3. O PDF reúne capa, folha de rosto, sumário, cinco páginas de conteúdo e referências.

O relatório aponta diretamente para a branch `feat/preprocessamento-etapa2`, que contém a versão revisada da análise. Essa branch ainda não foi integrada à `main`. A equipe autorizou tornar o repositório público para permitir o acesso da professora pelo link. Os CSVs e as cópias executadas locais continuam fora do Git.

## Cópia independente do notebook, opcional

```powershell
python preparar_entrega_etapa1.py
```

O comando gera `temp/entrega-etapa1/01_analise_exploratoria_entrega.ipynb`, `requirements.txt` e `LEIA-ME.md`. As funções atuais de `preprocessamento.py` são incluídas no notebook gerado; quem recebe a entrega não precisa desse módulo nem do repositório. A manutenção continua no notebook original e no módulo, para evitar duas implementações diferentes.

A geração produz uma cópia sem saídas: coloque o `treino.csv` oficial na mesma pasta, abra o notebook nessa pasta, selecione o Python com as bibliotecas indicadas e execute tudo a partir de um kernel reiniciado. Também é possível ajustar `DATA_PATH` para apontar ao treino em outro local. Essa cópia é um apoio para execução fora do repositório; o arquivo da entrega é o PDF. O CSV não é copiado pelo comando nem incluído no pacote de apoio. Os arquivos de teste não são necessários para a Etapa 1.

Requisitos: Python 3.11 ou superior, as versões de `requirements.txt` e um ambiente de notebooks com IPython. O ambiente usado na validação é Python 3.12. A cópia executada do notebook modular fica em `temp/validacao-etapa1-2026-11-01/`; a versão independente para entregar fica em `temp/entrega-etapa1/`.

## Escopo da Etapa 1

O notebook caracteriza base, desbalanceamento, ausências, categorias, duplicatas, conflitos, regras de domínio, `su_attempted`, redundâncias e extremos. Também investiga as relações de todos os atributos com a classe e o padrão de ausência das três colunas sob investigação temporal.

As associações estatísticas são evidências descritivas. A origem e a disponibilidade de `score_ids_legado`, `tipo_alerta_soc` e `tempo_resposta_analista_min` ainda precisam de confirmação para concluir se há vazamento. O notebook separa essas hipóteses dos fatos observados.

As regras de domínio são apresentadas separadamente dos limites indicativos dos contadores. As contagens de duplicatas e conflitos têm definições explícitas e não devem ser somadas. O impacto hipotético da remoção é mostrado por classe.

Nenhum modelo é treinado, nenhuma técnica de balanceamento é aplicada e nenhum registro é removido nesta etapa. A retirada de `num_outbound_cmds` das entradas dos futuros modelos, aprovada em 14/10, permanece documentada; o atributo original é mantido no diagnóstico.

## Registro de decisões

[DECISOES.md](DECISOES.md) reúne as escolhas aprovadas, as regras do enunciado e os pontos que continuam em discussão.

## Etapa 2 — Pré-processamento no Colab

No [Google Colab](https://colab.research.google.com/), use **Arquivo → Fazer upload de notebook** para abrir [02_preprocessamento_colab.ipynb](02_preprocessamento_colab.ipynb). Escolha **Ambiente de execução → Executar tudo** e forneça o `treino.csv` oficial do Canvas quando solicitado. A execução instala as bibliotecas necessárias em um ambiente separado quando as versões são diferentes. O notebook inclui todo o código e usa apenas esse CSV. A preparação das dependências e os 137 ajustes levam alguns minutos em CPU.

As tabelas mostram as alternativas de limpeza, imputação, codificação, escala, balanceamento e redução, além das combinações com árvore e rede neural. A validação usa divisões por grupo dentro do treino oficial. Os resultados e as justificativas de cada escolha estão em [DECISOES.md](DECISOES.md).

## Etapa 3 — Experimento inicial com SVM

O [notebook de SVM](03_svm_colab.ipynb) compara SVM linear e SVM com kernel RBF usando as divisões por grupo da Etapa 2. Abra no Colab, escolha **Executar tudo** e forneça o `treino.csv` oficial quando solicitado. O código executa em CPU. A variável `TRABALHADORES` permite rodar uma ou duas divisões em paralelo. Na validação local, a busca levou cerca de 11 minutos e a avaliação da quinta divisão levou mais cerca de 2 minutos.

O [notebook de SVM RBF em CPU e CUDA](03_svm_cuda_colab.ipynb) compara as tolerâncias `1e-3` e `1e-4` sem teto explícito de iterações. No Colab, selecione uma GPU em **Alterar tipo de ambiente de execução** e depois use **Executar tudo**. O notebook ajusta a SVM RBF com cuML nas quatro primeiras divisões e mede a CPU na primeira delas por padrão. Na sessão sem GPU, a comparação de tolerâncias roda em CPU.

O [notebook de Random Forest](03_random_forest_colab.ipynb) usa a mesma preparação e as mesmas divisões da SVM. Compara profundidade e ponderação de classes e apresenta a matriz de confusão e as métricas por classe na quinta divisão. No Colab, use **Executar tudo** e forneça o `treino.csv` oficial quando solicitado.

O [notebook de redes neurais](03_redes_neurais_colab.ipynb) compara uma MLP com 16 neurônios em uma camada oculta e uma MLP com três camadas ocultas de 64, 32 e 16 neurônios. Cada rede treina 120 épocas por divisão. A retropropagação calcula os gradientes e o Adam atualiza os pesos. As tabelas incluem as épocas, métricas e matrizes de confusão. O notebook roda em CPU no Colab e pede apenas o `treino.csv` oficial.

O [notebook de deep learning](03_deep_learning_colab.ipynb) treina uma candidata em PyTorch com quatro camadas ocultas e dropout. Ele mostra o ciclo de retropropagação e Adam e executa 120 épocas por divisão. No Colab, selecione GPU para usar CUDA ou execute em CPU. O único arquivo de dados necessário é o `treino.csv` oficial.
