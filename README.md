# NSL-KDD: Detecção de Intrusão em Redes

Projeto acadêmico de Inteligência Artificial para investigar o pré-processamento e comparar classificadores na base NSL-KDD.

O desenvolvimento é realizado com decisões discutidas e justificadas em conjunto. O projeto está na etapa de diagnóstico dos dados de treino; a definição das classes e as técnicas finais permanecem em discussão.

## Base de dados

Fonte indicada no enunciado: [NSL-KDD no Kaggle](https://www.kaggle.com/datasets/hassan06/nslkdd).

Os arquivos da base são mantidos localmente em `data/raw/` e ignorados pelo Git. Para baixar o ZIP no Windows, a partir desta pasta:

```powershell
New-Item -ItemType Directory -Path data/raw -Force
curl.exe --fail --location --output data/raw/nslkdd.zip https://www.kaggle.com/api/v1/datasets/download/hassan06/nslkdd
```

Documentação original: [Canadian Institute for Cybersecurity / UNB](https://www.unb.ca/cic/datasets/nsl.html).

Referência da base: M. Tavallaee, E. Bagheri, W. Lu e A. Ghorbani, *A Detailed Analysis of the KDD CUP 99 Data Set*, CISDA, 2009.

## Organização inicial

- `01_analise_exploratoria.ipynb`: diagnóstico do treino e comparação do efeito da limpeza dos conflitos.
- `preprocessamento.py`: leitura da base, seleção das entradas e funções de identificação e tratamento dos conflitos.
- `requirements.txt`: versão das dependências usadas nesta etapa.
- `tests/`: verificações de preservação dos dados, alinhamento dos rótulos e isolamento do treino.
- `data/raw/`: base original local, fora do versionamento.
- `temp/`: documentos de diagnóstico, fora do versionamento.
- `conflitos-treino.csv`: amostra local exportada para inspeção, fora do versionamento.

O registro permanente das decisões será preparado em uma próxima etapa.

## Executar esta etapa

Use um ambiente de notebooks Python e instale as dependências:

```powershell
python -m pip install -r requirements.txt
```

Depois abra `01_analise_exploratoria.ipynb` e execute as células na ordem. O ZIP deve estar no caminho descrito acima. A primeira versão descreve os rótulos originais e compara cenários de diagnóstico; o alvo oficial continua pendente.

Para verificar as funções:

```powershell
python -m unittest discover -s tests -v
```

O notebook é versionado sem saídas com registros da base. Ao salvar uma versão executada, limpe as saídas antes de enviá-la ao GitHub.

## Cuidados metodológicos

- Decisão de 14/10/2026: excluir `num_outbound_cmds` das entradas dos futuros modelos, pois é zero em todo o treino. A função `selecionar_entradas` retorna 40 atributos; a base original e o diagnóstico mantêm os 41.
- Preservar a separação oficial entre `KDDTrain+` e `KDDTest+`.
- Reservar o conjunto de teste para a avaliação final.
- Ajustar o pré-processamento e os hiperparâmetros somente com os dados de treinamento de cada rodada de validação.
- Comparar os experimentos usando F1-macro, com atenção às classes raras.

As entregas previstas são um notebook comentado e reprodutível, um relatório com as justificativas e comparações e a apresentação dos resultados.
