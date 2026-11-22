"""Sincroniza as cópias independentes do preparo nas etapas 2 e 3.

Uso: python preparar_notebooks_etapa3.py [--check] [--notebook NOME.ipynb]
Os notebooks continuam executáveis no Colab sem arquivos auxiliares.
"""

import argparse
import json
from pathlib import Path


RAIZ = Path(__file__).resolve().parent
NOTEBOOKS = (
    "02_preprocessamento_colab.ipynb",
    "03_random_forest_colab.ipynb",
    "03_redes_neurais_colab.ipynb",
    "03_deep_learning_colab.ipynb",
    "03_svm_colab.ipynb",
    "03_svm_cuda_colab.ipynb",
)
IMPORTS_ETAPA2 = """import warnings
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

"""


def sincronizar(nome: str, *, check: bool = False) -> bool:
    """Retorna True se o notebook já coincidia com as fontes."""
    if nome not in NOTEBOOKS:
        raise ValueError(f"Notebook não previsto: {nome}")
    caminho = RAIZ / nome
    original = caminho.read_text(encoding="utf-8")
    notebook = json.loads(original)
    fontes = {
        "definicoes-dados": ("{codigo_dados}", "dados_etapa2.py"),
        "definicoes-tecnicas": ("{codigo_tecnicas}", "tecnicas_etapa2.py"),
    }
    for identificador, (variavel, arquivo) in fontes.items():
        celulas = [c for c in notebook["cells"] if c.get("id") == identificador]
        if len(celulas) != 1:
            raise ValueError(f"Esperada uma célula {identificador} em {nome}")
        celula = celulas[0]
        atual = "".join(celula["source"])
        cabecalho = f'%%writefile "{variavel}"\n'
        if not atual.startswith(cabecalho):
            raise ValueError(f"Cabeçalho inesperado em {nome}:{identificador}")
        fonte = (RAIZ / arquivo).read_text(encoding="utf-8").rstrip() + "\n"
        if nome.startswith("02_") and identificador == "definicoes-tecnicas":
            # A rotina de comparação da Etapa 2 é específica desse notebook.
            # Preservá-la depois do bloco compartilhado evita mudar resultados.
            marcador = "def medir("
            if marcador not in atual:
                raise ValueError("A rotina medir da Etapa 2 não foi encontrada")
            exclusivo = atual[atual.index(marcador):]
            fonte = IMPORTS_ETAPA2 + fonte + "\n" + exclusivo
        celula["source"] = (cabecalho + fonte).splitlines(keepends=True)
    novo = json.dumps(notebook, ensure_ascii=False, indent=2) + "\n"
    if novo != original and not check:
        caminho.write_text(novo, encoding="utf-8")
    return novo == original


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verifica sem escrever")
    parser.add_argument("--notebook", choices=NOTEBOOKS)
    argumentos = parser.parse_args()
    nomes = (argumentos.notebook,) if argumentos.notebook else NOTEBOOKS
    diferentes = []
    for nome in nomes:
        igual = sincronizar(nome, check=argumentos.check)
        print(f"{nome}: {'sincronizado' if igual else 'atualizado' if not argumentos.check else 'divergente'}")
        if not igual:
            diferentes.append(nome)
    if argumentos.check and diferentes:
        parser.exit(1, "Cópias divergentes: " + ", ".join(diferentes) + "\n")


if __name__ == "__main__":
    main()
