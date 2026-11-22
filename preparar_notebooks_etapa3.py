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
    "03_comparacao_modelos_colab.ipynb",
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
    if nome == "03_comparacao_modelos_colab.ipynb":
        return sincronizar_comparacao(check=check)
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


def sincronizar_comparacao(*, check=False):
    """Monta uma copia independente a partir das mesmas fontes do executor local."""
    celulas = []

    def adicionar(tipo, identificador, texto):
        c = {"cell_type": tipo, "id": identificador, "metadata": {},
             "source": texto.strip().splitlines(keepends=True)}
        if tipo == "code":
            c.update(execution_count=None, outputs=[])
        celulas.append(c)

    adicionar("markdown", "introducao", """
# Etapa 3 — Comparação dos classificadores NSL-KDD

**Disciplina:** Inteligência Artificial · Ciência da Computação · PUC Minas

**Equipe:** Trabalho Prático 3 (nomes dos integrantes a completar pela equipe).

O objetivo é comparar classificadores de intrusão usando somente o **treino.csv oficial**,
com `class` como alvo (`normal` ou `anomaly`). A base contém 69.833 registros e 49 atributos;
as contagens e o esquema são conferidos novamente na execução. F1-macro é o critério
principal, com classes na ordem **normal, anomaly**. O teste sem rótulos não é usado.

Os resultados históricos de `DECISOES.md` (incluindo escolha antiga de épocas e RF ≈ 0,8744)
são referências de outro protocolo e não entram nas médias desta execução.
A quinta divisão histórica foi reutilizada: nenhum resultado aqui é chamado de teste intocado.

No Colab, execute todas as células em CPU. O orçamento é de até 210 ajustes sequenciais,
com a base completa e redes de 120 épocas. Reserve aproximadamente 15–60 minutos;
a duração depende da máquina. A cada ajuste, os resultados são persistidos.
Para sobreviver ao encerramento da sessão Colab, monte seu Drive e configure
`NSLKDD_DESTINO` para uma pasta persistente **antes de executar**. O destino padrão
fica no disco da sessão. Não publique essa pasta: ela contém índices e previsões por linha.
""")
    resumo_salvo = RAIZ / "output/stage3/resultados.json"
    if resumo_salvo.exists():
        registro = json.loads(resumo_salvo.read_text(encoding="utf-8"))
        selecao = registro["selecao"]
        texto = ["## Resultados observados no treino oficial", "",
                 "Este resumo agregado permite revisar os resultados sem novo treinamento. "
                 "As células abaixo reproduzem a execução e geram suas próprias tabelas e gráficos.", "",
                 f"Dados SHA-256: `{registro['sha256_dados']}`. "
                 f"Tipo: `{registro['tipo_dados']}`; cinco folds externos e dois internos.", "",
                 "| Comparação | F1-macro médio | Desvio | Precisão anomaly | Recall anomaly |",
                 "| --- | ---: | ---: | ---: | ---: |"]
        for r in registro["resumos"]:
            texto.append(f"| {r['comparacao']} | {r['f1_macro_media']:.6f} | "
                         f"{r['f1_macro_desvio']:.6f} | {r['precision_anomaly_media']:.6f} | "
                         f"{r['recall_anomaly_media']:.6f} |")
        texto.extend(["", f"**Pipeline recomendado:** `{selecao['configuracao_recomendada']}`. "
                      "A escolha usa F1-macro médio. A RF com oversampling obteve 0,878817 "
                      "e a RF ponderada, 0,878042: a diferença de 0,000775 é pequena e não "
                      "estabelece superioridade estatística. Desvios, erros por classe, gap "
                      "treino–validação e custo constam das tabelas e figuras abaixo. " + selecao["limite"]])
        adicionar("markdown", "resultado-registrado", "\n".join(texto))
    adicionar("code", "ambiente", '''
import json, os, subprocess, sys
from pathlib import Path
from IPython.display import display, Markdown, Image

if sys.version_info < (3, 12):
    raise RuntimeError("Use um runtime Python 3.12 ou posterior compatível com os pins.")
BASE = Path("temp/stage3_notebook").resolve()
CODIGO = BASE / "codigo"
CODIGO.mkdir(parents=True, exist_ok=True)
DESTINO = Path(os.environ.get("NSLKDD_DESTINO", "temp/stage3_runs/comparacao_cpu")).resolve()
SAIDA = DESTINO / "resumo"
AMBIENTE = os.environ.copy()
AMBIENTE.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
''')
    pins = (RAIZ / "requirements-stage3-cpu.txt").read_text(encoding="utf-8")
    pins = [s.strip() for s in pins.splitlines() if s.strip() and not s.startswith("#")]
    pins += ["matplotlib==3.10.8"]
    adicionar("code", "dependencias", f'''
PINS = {pins!r}
esperadas = dict(p.split("==") for p in PINS)
bibliotecas = BASE / "bibliotecas"
AMBIENTE["PYTHONPATH"] = os.pathsep.join([str(CODIGO), str(bibliotecas)])
consulta = "import json; from importlib.metadata import version; print(json.dumps({{p: version(p) for p in " + repr(list(esperadas)) + "}}))"
def consultar():
    r = subprocess.run([sys.executable, "-c", consulta], env=AMBIENTE, capture_output=True, text=True)
    return json.loads(r.stdout) if r.returncode == 0 else {{}}
if consultar() != esperadas:
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "--target", str(bibliotecas), *PINS], check=True)
if consultar() != esperadas:
    raise RuntimeError("As versões instaladas divergem das versões de referência.")
print(consultar())
''')
    adicionar("markdown", "fontes-nota", """
## Preparação e protocolo

As células seguintes escrevem os módulos mantidos no projeto. A divisão agrupada usa entradas
iguais na representação C como grupo, inclusive quando há rótulos conflitantes.
São cinco folds externos e dois internos em cada treino externo, sempre com semente 42.
Imputação, vocabulário, escala, PCA, deduplicação e amostragem são ajustados somente no treino
correspondente. A validação mantém todas as linhas. As comparações compartilham os folds externos.

RF compara profundidade 12/livre e pesos ausentes/`balanced_subsample`, com 100 árvores.
SVM compara linear C=0,1 e RBF C=10; somente ajustes convergidos são elegíveis.
MLPs usam (16,) e (64,32,16), Adam, oversampling 0,5 e 120 épocas fixas.
PyTorch e CUDA são opcionais e não fazem parte deste orçamento.

Na RF livre sem ponderação, variamos um fator por vez: over/undersampling, ordinal/one-hot,
escala/ausência de escala, deduplicação/todas as linhas, redundâncias A/C e PCA até 60 componentes.
Regras de inconsistência e exclusões aprovadas permanecem em todos os candidatos;
o CSV original nunca é alterado.
""")
    for nome in ("dados_etapa2.py", "tecnicas_etapa2.py", "stage3_evaluation.py", "stage3_comparison.py"):
        # Preserva bytes e hash, inclusive finais de linha, entre Windows e Colab.
        fonte = (RAIZ / nome).read_bytes()
        adicionar("code", "fonte-" + nome.replace(".py", "").replace("_", "-"),
                  f"(CODIGO / {nome!r}).write_bytes({fonte!r})")
        celulas[-1]["metadata"] = {"jupyter": {"source_hidden": True}}
    adicionar("code", "leitura-base", '''
candidatos = [Path(os.environ["NSLKDD_TREINO"])] if "NSLKDD_TREINO" in os.environ else [Path("data/raw/treino.csv"), Path("treino.csv")]
DATA_PATH = next((p.resolve() for p in candidatos if p.is_file()), None)
if DATA_PATH is None:
    try:
        from google.colab import files
    except ImportError:
        raise FileNotFoundError("Disponibilize data/raw/treino.csv ou configure NSLKDD_TREINO.")
    enviados = files.upload()
    import re
    if len(enviados) != 1 or not re.fullmatch(r"treino(?: \\(\\d+\\))?\\.csv", next(iter(enviados))):
        raise ValueError("Selecione somente o treino.csv oficial (também aceita treino (1).csv).")
    DATA_PATH = Path(next(iter(enviados))).resolve()
if DATA_PATH.name != "treino.csv":
    # Downloads repetidos podem acrescentar (1); a copia preserva todos os bytes.
    entrada = BASE / "entrada"
    entrada.mkdir(exist_ok=True)
    copia = entrada / "treino.csv"
    copia.write_bytes(DATA_PATH.read_bytes())
    DATA_PATH = copia
codigo = "from stage3_comparison import verificar_base; import json,sys; print(json.dumps(verificar_base(sys.argv[1])))"
r = subprocess.run([sys.executable, "-c", codigo, str(DATA_PATH)], env=AMBIENTE, check=True, capture_output=True, text=True)
diagnostico = json.loads(r.stdout)
display(Markdown(f"**Base:** {diagnostico['registros']} registros, {diagnostico['atributos']} atributos. Classes: {diagnostico['classes']}. SHA-256: `{diagnostico['sha256']}`."))
''')
    adicionar("code", "executar", '''
# Reexecutar com o mesmo destino retoma apenas ajustes compatíveis e completos.
# O subprocesso usa as dependências verificadas, sem trocar pacotes do kernel.
subprocess.run([sys.executable, "-u", str(CODIGO / "stage3_comparison.py"),
                "--dados", str(DATA_PATH), "--destino", str(DESTINO), "--saida", str(SAIDA)],
               env=AMBIENTE, check=True)
''')
    adicionar("markdown", "resultados-nota", """
## Resultados e comparações

As tabelas abaixo são recalculadas das previsões registradas. Todos os folds são mostrados,
incluindo diagnósticos inelegíveis. O ranking exige cobertura completa e convergência.
O desvio é amostral entre folds; não é intervalo de confiança e não prova significância.
As métricas agregadas dos gráficos de matriz somam previsões externas: cada linha aparece uma vez.
O F1 médio dos folds é diferente do F1 calculado sobre essa matriz somada.
""")
    adicionar("code", "tabelas", '''
import csv, html
from IPython.display import HTML
for nome in ("comparacao_modelos.csv", "resultados_por_divisao.csv", "configuracoes_internas.csv"):
    with (SAIDA / nome).open(encoding="utf-8", newline="") as arquivo:
        linhas = list(csv.reader(arquivo))
    display(Markdown("### " + nome))
    display(HTML("<div style='overflow:auto;max-height:500px'><table>" + "".join(
        "<tr>" + "".join("<td>" + html.escape(v) + "</td>" for v in linha) + "</tr>" for linha in linhas) + "</table></div>"))
''')
    adicionar("code", "graficos", '''
for nome in ("f1_algoritmos.png", "precisao_recall.png", "matrizes_confusao.png", "variacao_divisoes.png", "efeito_preprocessamento.png"):
    display(Image(filename=str(SAIDA / nome)))
''')
    adicionar("code", "conclusoes", '''
resultado = json.loads((SAIDA / "resultados.json").read_text(encoding="utf-8"))
selecao = resultado["selecao"]
display(Markdown(f"**Comparação selecionada:** {selecao['comparacao_selecionada']}. "
                 f"**Configuração recomendada:** {selecao['configuracao_recomendada']}."))
display(Markdown(selecao["criterio"] + " " + selecao["limite"]))
selecionada = next((c for c in resultado["configuracoes"] if c["nome"] == selecao["configuracao_recomendada"]), None)
print(json.dumps(selecionada, ensure_ascii=False, indent=2))
''')
    adicionar("markdown", "limites", """
## Justificativa e conclusões da Etapa 3

A recomendação usa F1-macro médio; precisão/recall de anomalias, falsos positivos/negativos,
gap treino-validação, desvio e tempo nas tabelas permitem avaliar o custo da escolha.
A comparação externa participa da seleção final: o desempenho do vencedor pode ser otimista.
`selecao_global_aninhada` mede separadamente o procedimento que escolhe candidatos somente
pelos folds internos; tampouco elimina o conhecimento histórico da base.

O diagnóstico não altera épocas ou hiperparâmetros após observar os resultados externos.
O treino integral do pipeline para submissão, inferência no teste, explicabilidade e relatório
acadêmico final pertencem a etapas futuras. Permanecem pendentes a proveniência dos três
sinais excluídos e o significado de `su_attempted=2`. Os resultados são específicos desta base,
semente e orçamento; não demonstram desempenho sobre tráfego contemporâneo.

**Declaração de uso de IA generativa.** O Codex auxiliou na organização do código,
na construção do notebook, na execução dos experimentos, na conferência dos registros
e na redação técnica. Os dados, parâmetros, métricas e limitações apresentados são
verificáveis pelas células e pelos resultados registrados; a equipe é responsável pela
revisão e interpretação acadêmica do trabalho.
""")
    notebook = {"cells": celulas, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                                "language_info": {"name": "python", "version": "3.12"}},
                "nbformat": 4, "nbformat_minor": 5}
    caminho = RAIZ / "03_comparacao_modelos_colab.ipynb"
    novo = json.dumps(notebook, ensure_ascii=False, indent=2) + "\n"
    igual = caminho.exists() and caminho.read_text(encoding="utf-8") == novo
    if not check and not igual:
        caminho.write_text(novo, encoding="utf-8")
    return igual


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
