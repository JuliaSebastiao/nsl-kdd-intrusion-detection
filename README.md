# NSL-KDD: Detecção de Intrusão em Redes

Projeto acadêmico da competição de IA. A primeira entrega é uma análise exploratória reprodutível da base oficial do Canvas.

## Base oficial

Coloque os arquivos oficiais do Canvas em `data/raw/`: `treino.csv`, `teste_sem_rotulo.csv` e `modelo_de_submissao.csv`. A pasta e esses nomes de arquivo estão no `.gitignore`; os dados não devem ser versionados. O notebook da Etapa 1 lê apenas `data/raw/treino.csv`, usa `class` como alvo e não depende de Kaggle, internet, `nslkdd.zip`, `KDDTrain+.txt`, `attack_type` ou `difficulty`.

## Executar

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt ipykernel
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Abra `01_analise_exploratoria.ipynb` a partir da raiz, selecione o Python da `.venv` no ambiente de notebooks e execute as células na ordem. Quem já usa outro ambiente de notebooks pode instalar `requirements.txt` nesse ambiente.

O notebook do Git é mantido sem saídas com registros da base. Para entregar, salve uma cópia executada em `temp/`, que permanece fora do versionamento.

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
