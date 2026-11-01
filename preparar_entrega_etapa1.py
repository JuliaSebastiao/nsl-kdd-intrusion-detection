"""Prepara uma cópia de entrega sem depender do módulo local ou incluir a base.

Uso: python preparar_entrega_etapa1.py
Depois, abra a cópia gerada com o treino oficial na mesma pasta e execute tudo.
"""

import argparse
import hashlib
import json
from pathlib import Path


RAIZ = Path(__file__).resolve().parent


def preparar_entrega(destino: Path) -> Path:
    """Inclui as funções atuais no notebook e copia as instruções de execução."""
    notebook = json.loads((RAIZ / "01_analise_exploratoria.ipynb").read_text(encoding="utf-8"))
    modulo = (RAIZ / "preprocessamento.py").read_text(encoding="utf-8")
    indices = [i for i, c in enumerate(notebook["cells"])
               if c["cell_type"] == "code" and "from preprocessamento import *" in "".join(c["source"])]
    if len(indices) != 1:
        raise ValueError("Esperada exatamente uma célula de carregamento do módulo local.")
    indice = indices[0]
    instrucoes = """### Como reproduzir esta entrega

As funções usadas na análise estão incluídas na célula seguinte. Não é necessário ter o repositório ou `preprocessamento.py`.

1. Use Python 3.11 ou superior e instale as versões de `requirements.txt`. O ambiente validado usa Python 3.12.
2. Coloque o `treino.csv` oficial na mesma pasta do notebook e inicie o notebook nessa pasta. Se o arquivo estiver em outro local, altere somente `DATA_PATH` na célula de carregamento.
3. Selecione o kernel desse ambiente e execute todas as células, na ordem, a partir de um kernel reiniciado.

Dependências da análise: pandas 3.0.1, NumPy 2.4.2, Matplotlib 3.10.8 e Seaborn 0.13.2. É necessário um ambiente de notebooks com IPython. Não é necessário baixar outros dados nem fornecer o teste.
"""
    notebook["cells"].insert(indice, {
        "cell_type": "markdown", "metadata": {}, "id": "entrega-instrucoes",
        "source": instrucoes.splitlines(keepends=True),
    })
    notebook["cells"].insert(indice + 1, {
        "cell_type": "code", "metadata": {}, "id": "entrega-funcoes",
        "execution_count": None, "outputs": [],
        "source": modulo.splitlines(keepends=True),
    })
    carregamento = """%matplotlib inline
from pathlib import Path
from IPython.display import Markdown, display
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Única configuração de caminho: use o arquivo oficial de treino.
DATA_PATH = Path("treino.csv")
dados = carregar_treino(DATA_PATH)
print(DATA_PATH.name, dados.shape)
"""
    notebook["cells"][indice + 2]["source"] = carregamento.splitlines(keepends=True)
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        elif cell["cell_type"] == "markdown":
            texto = "".join(cell["source"]).replace("`data/raw/treino.csv`", "`treino.csv`")
            cell["source"] = texto.splitlines(keepends=True)
    notebook["metadata"]["kernelspec"] = {
        "display_name": "Python 3", "language": "python", "name": "python3",
    }
    notebook["metadata"]["entrega_etapa1"] = {
        "notebook_origem": "01_analise_exploratoria.ipynb",
        "funcoes_origem": "preprocessamento.py",
        "sha256_funcoes": hashlib.sha256(modulo.encode("utf-8")).hexdigest(),
    }
    destino.mkdir(parents=True, exist_ok=True)
    target = destino / "01_analise_exploratoria_entrega.ipynb"
    target.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (destino / "requirements.txt").write_text((RAIZ / "requirements.txt").read_text(encoding="utf-8"), encoding="utf-8")
    (destino / "LEIA-ME.md").write_text(
        "# Entrega da Etapa 1\n\n"
        "O notebook inclui as funções de diagnóstico e dispensa `preprocessamento.py`. "
        "A versão executada contém as tabelas, os gráficos e o diagnóstico breve.\n\n"
        "## Reproduzir\n\n"
        "1. Use Python 3.11 ou superior; a validação foi feita com Python 3.12.\n"
        "2. Instale as bibliotecas: `python -m pip install -r requirements.txt ipykernel`.\n"
        "3. Coloque o `treino.csv` oficial na mesma pasta do notebook. O CSV não acompanha a entrega.\n"
        "4. Abra no VS Code, Jupyter ou outro ambiente de notebooks, selecione o Python com essas bibliotecas, "
        "reinicie o kernel e execute todas as células em ordem. Inicie o notebook nessa pasta ou ajuste `DATA_PATH`.\n\n"
        "Não é necessário fornecer o teste, acessar o repositório ou buscar a base no Kaggle. "
        "A análise não altera o CSV. O notebook do projeto e o módulo continuam sendo as fontes de manutenção; "
        "esta cópia é gerada por `preparar_entrega_etapa1.py`.\n\n"
        "## Arquivos\n\n"
        "- `01_analise_exploratoria_entrega.ipynb`: notebook para a entrega.\n"
        "- `requirements.txt`: versões das bibliotecas usadas na análise.\n"
        "- `LEIA-ME.md`: estas instruções.\n",
        encoding="utf-8",
    )
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destino", type=Path, default=RAIZ / "temp" / "entrega-etapa1")
    args = parser.parse_args()
    target = preparar_entrega(args.destino.resolve())
    print(f"Cópia preparada, ainda sem executar: {target}")


if __name__ == "__main__":
    main()
