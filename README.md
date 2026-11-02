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

## Preparação da Etapa 2

As exclusões aprovadas estão em `COLUNAS_EXCLUIDAS_EXPERIMENTOS`: `num_outbound_cmds` e, provisoriamente, `tempo_resposta_analista_min`, `tipo_alerta_soc` e `score_ids_legado`. A leitura e o diagnóstico preservam essas colunas. Foi aprovada a comparação de três representações: manter as cinco derivadas (45 atributos candidatos), retirar as quatro cópias mantendo `total_bytes` (41), ou retirar as cinco derivadas (40). As contagens são anteriores à codificação; nenhuma versão foi escolhida como vencedora.

As entradas nominais são diagnosticadas na Etapa 1, mas ainda não foram codificadas para modelos. `protocolo_cod` é nominal apesar dos códigos numéricos. One-hot encoding é a proposta inicial em discussão para a Etapa 2, que exige pelo menos uma técnica de codificação justificada. A proposta e os cuidados com ausências, categorias novas e ajuste apenas no treino estão em [DECISOES.md](DECISOES.md).
