# NSL-KDD: Detecção de Intrusão em Redes

Projeto acadêmico da competição de IA. A primeira entrega é uma análise exploratória reprodutível da base oficial do Canvas.

## Base oficial

Coloque `treino.csv` na raiz deste projeto. O arquivo não deve ser versionado nem redistribuído se as regras do curso não permitirem. O notebook usa `class` como alvo e não depende de Kaggle, internet, `nslkdd.zip`, `KDDTrain+.txt`, `attack_type` ou `difficulty`.

## Executar

```powershell
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Abra `01_analise_exploratoria.ipynb` a partir da raiz e execute as células na ordem.

## Escopo da Etapa 1

O notebook caracteriza base, desbalanceamento, ausências, categorias, duplicatas, conflitos, regras de domínio, `su_attempted`, redundâncias e extremos. Ele não treina modelos, não balanceia dados e não remove registros. A investigação de possível leakage permanece reservada para integração posterior.
