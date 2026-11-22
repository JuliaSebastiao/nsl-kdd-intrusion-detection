"""Infraestrutura da Etapa 3: selecao interna e avaliacao externa por grupos.

Nenhum treinamento roda ao importar este modulo. Resultados oficiais exigem
explicitamente o caminho do treino.csv; fixtures recebem tipo sintético.
"""

from dataclasses import asdict, dataclass, field
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter

from imblearn.over_sampling import RandomOverSampler
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC, SVC

from dados_etapa2 import SEMENTE, carregar_treino, preparar_base
from tecnicas_etapa2 import transformar


CLASSES = ("normal", "anomaly")
PACOTES = ("numpy", "pandas", "scipy", "scikit-learn", "imbalanced-learn",
           "joblib", "threadpoolctl", "sklearn-compat", "torch", "cuml")


@dataclass(frozen=True)
class Configuracao:
    nome: str
    modelo: str
    parametros: dict = field(default_factory=dict)
    linhas: str = "sem_duplicatas"
    numericos: str = "mediana"
    nominais: str = "ausencia"
    codificacao: str = "onehot"
    escala: str = "zscore"
    balanceamento: str = "nenhum"


CONFIGURACOES_REFERENCIA = (
    Configuracao("rf_livre_ponderada", "random_forest",
                 {"n_estimators": 100, "max_depth": None, "class_weight": "balanced_subsample",
                  "min_samples_leaf": 2, "max_features": "sqrt", "criterion": "gini"}),
    Configuracao("svm_rbf_c10", "svm_rbf", {"C": 10.0, "gamma": "scale"}),
    Configuracao("svm_linear_c01", "svm_linear", {"C": 0.1}),
    Configuracao("mlp_pequena_120", "mlp_pequena", {"epocas": 120}, balanceamento="over_0_5"),
    Configuracao("mlp_profunda_120", "mlp_profunda", {"epocas": 120}, balanceamento="over_0_5"),
    Configuracao("pytorch_profunda_120", "pytorch_profunda", {"epocas": 120, "dispositivo": "cpu"},
                 balanceamento="over_0_5"),
)


def _digest_indices(indices):
    return sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest()


def planejar_divisoes(dados, *, externas=5, internas=3, semente=SEMENTE):
    """Constroi folds externos; cada treino externo recebe folds internos proprios."""
    representacoes, grupos = preparar_base(dados)
    x = representacoes["C"]
    y = dados["class"].to_numpy()
    if externas < 2 or internas < 2:
        raise ValueError("Sao necessarias pelo menos duas divisoes em cada nivel.")
    outer = StratifiedGroupKFold(n_splits=externas, shuffle=True, random_state=semente)
    plano = []
    for numero, (treino, validacao) in enumerate(outer.split(x, y, grupos), 1):
        inner = StratifiedGroupKFold(n_splits=internas, shuffle=True, random_state=semente)
        interno = []
        for subtreino, subvalidacao in inner.split(x.iloc[treino], y[treino], grupos.iloc[treino]):
            ajuste, val_interna = treino[subtreino], treino[subvalidacao]
            if set(grupos.iloc[ajuste]) & set(grupos.iloc[val_interna]):
                raise AssertionError("Grupos sobrepostos na selecao interna")
            interno.append((ajuste, val_interna))
        if set(grupos.iloc[treino]) & set(grupos.iloc[validacao]):
            raise AssertionError("Grupos sobrepostos na avaliacao externa")
        plano.append((numero, treino, validacao, interno))
    return x, y, grupos, plano


class _MLPExistente:
    """Mantem o treino por partial_fit e as arquiteturas dos notebooks atuais."""

    def __init__(self, camadas, epocas):
        self.epocas = epocas
        self.modelo = MLPClassifier(
            hidden_layer_sizes=camadas, activation="relu", solver="adam", batch_size=512,
            alpha=0.0001, learning_rate_init=0.001, shuffle=True,
            random_state=np.random.RandomState(SEMENTE), early_stopping=False,
            max_iter=epocas + 1, n_iter_no_change=epocas + 1,
        )

    def fit(self, x, y):
        rotulos = (np.asarray(y) == "anomaly").astype(np.int32)
        for _ in range(self.epocas):
            self.modelo.partial_fit(x.astype(np.float32), rotulos, classes=[0, 1])
        self.epocas_concluidas_ = len(self.modelo.loss_curve_)
        return self

    def predict(self, x):
        previsto = self.modelo.predict(x.astype(np.float32))
        return np.where(previsto == 1, "anomaly", "normal")


class _PyTorchExistente:
    """Adapter opcional da rede profunda, com as camadas do notebook PyTorch."""

    def __init__(self, epocas, dispositivo):
        self.epocas = epocas
        self.dispositivo = dispositivo

    def fit(self, x, y):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        import torch
        from torch import nn

        if self.dispositivo == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA indisponivel no PyTorch deste ambiente")
        device = torch.device(self.dispositivo)
        torch.manual_seed(SEMENTE)
        torch.use_deterministic_algorithms(True)
        gerador = torch.Generator(device=device).manual_seed(SEMENTE)
        camadas = []
        entradas = x.shape[1]
        for indice, saidas in enumerate((128, 64, 32, 16)):
            camadas.extend((nn.Linear(entradas, saidas), nn.ReLU()))
            if indice < 3:
                camadas.append(nn.Dropout(0.15))
            entradas = saidas
        camadas.append(nn.Linear(entradas, 1))
        self.modelo = nn.Sequential(*camadas).to(device)
        entradas_t = torch.as_tensor(np.asarray(x, dtype=np.float32, order="C"), device=device)
        rotulos_t = torch.as_tensor((np.asarray(y) == "anomaly").astype(np.float32), device=device)
        perda = nn.BCEWithLogitsLoss()
        otimizador = torch.optim.Adam(self.modelo.parameters(), lr=0.001, weight_decay=0.0001)
        self.modelo.train()
        for _ in range(self.epocas):
            indices = torch.randperm(len(rotulos_t), generator=gerador, device=device)
            for lote in indices.split(512):
                otimizador.zero_grad(set_to_none=True)
                valor = perda(self.modelo(entradas_t[lote]).squeeze(1), rotulos_t[lote])
                valor.backward()
                otimizador.step()
        self.epocas_concluidas_ = self.epocas
        return self

    def predict(self, x):
        import torch

        self.modelo.eval()
        device = next(self.modelo.parameters()).device
        entradas = torch.as_tensor(np.asarray(x, dtype=np.float32, order="C"), device=device)
        with torch.inference_mode():
            previsto = (self.modelo(entradas).squeeze(1) >= 0).cpu().numpy()
        return np.where(previsto, "anomaly", "normal")


def criar_modelo(config):
    parametros = dict(config.parametros)
    if config.modelo == "random_forest":
        return RandomForestClassifier(random_state=SEMENTE, n_jobs=1, **parametros)
    if config.modelo == "svm_linear":
        return LinearSVC(dual=False, max_iter=5000, random_state=SEMENTE, **parametros)
    if config.modelo == "svm_rbf":
        return SVC(kernel="rbf", cache_size=256, max_iter=200_000, **parametros)
    if config.modelo in ("mlp_pequena", "mlp_profunda"):
        camadas = (16,) if config.modelo == "mlp_pequena" else (64, 32, 16)
        return _MLPExistente(camadas, parametros.get("epocas", 120))
    if config.modelo == "pytorch_profunda":
        return _PyTorchExistente(parametros.get("epocas", 120), parametros.get("dispositivo", "cpu"))
    raise ValueError(f"Modelo desconhecido: {config.modelo}")


def status_convergencia(modelo, familia):
    """Retorna (estado, elegivel); ausencia de status nunca prova convergencia."""
    if familia == "svm_linear":
        iteracoes = getattr(modelo, "n_iter_", None)
        limite = getattr(modelo, "max_iter", None)
        if iteracoes is None or limite is None:
            return "desconhecido", False
        elegivel = bool(np.all(np.asarray(iteracoes) < limite))
        return ("convergiu" if elegivel else "limite_de_iteracoes"), elegivel
    if familia == "svm_rbf":
        estado = getattr(modelo, "fit_status_", None)
        if estado is None:
            return "desconhecido", False
        return ("convergiu" if estado == 0 else f"fit_status_{estado}"), bool(estado == 0)
    if familia.startswith("mlp_") or familia == "pytorch_profunda":
        concluidas = getattr(modelo, "epocas_concluidas_", None)
        elegivel = concluidas == getattr(modelo, "epocas", None) and concluidas is not None
        return ("epocas_completas" if elegivel else "epocas_incompletas"), bool(elegivel)
    return "ajuste_completo", True


def calcular_metricas(y_real, y_previsto):
    real, previsto = np.asarray(y_real), np.asarray(y_previsto)
    if real.shape != previsto.shape or real.ndim != 1:
        raise ValueError("Rotulos reais e previstos devem ter o mesmo formato")
    if not set(real).issubset(CLASSES) or not set(previsto).issubset(CLASSES):
        raise ValueError("Rotulos fora de normal/anomaly")
    relatorio = classification_report(real, previsto, labels=CLASSES,
                                       output_dict=True, zero_division=0)
    return {
        "f1_macro": float(f1_score(real, previsto, labels=CLASSES, average="macro", zero_division=0)),
        "acuracia": float(accuracy_score(real, previsto)),
        "matriz": confusion_matrix(real, previsto, labels=CLASSES).tolist(),
        "por_classe": {classe: {chave: float(relatorio[classe][chave])
                                 for chave in ("precision", "recall", "f1-score", "support")}
                       for classe in CLASSES},
    }


def avaliar_divisao(x, y, ajuste, validacao, config, *, fabrica=criar_modelo):
    """Mede um pipeline sem ajustar transformacoes nem amostragem na validacao."""
    ajuste = np.asarray(ajuste, dtype=int)
    validacao = np.asarray(validacao, dtype=int)
    if set(ajuste) & set(validacao):
        raise ValueError("Ajuste e validacao se sobrepoem")
    inicio = perf_counter()
    x_tr, x_val, y_tr = transformar(x, y, ajuste, validacao, config)
    segundos_preparo = perf_counter() - inicio
    if not np.isfinite(x_tr).all() or not np.isfinite(x_val).all():
        raise ValueError("Transformacao produziu valores nao finitos")
    x_treino_original, y_treino_original = x_tr, y_tr
    if config.balanceamento == "over_0_5":
        x_tr, y_tr = RandomOverSampler(sampling_strategy=0.5, random_state=SEMENTE).fit_resample(x_tr, y_tr)
    elif config.balanceamento != "nenhum":
        raise ValueError(f"Balanceamento desconhecido: {config.balanceamento}")
    modelo = fabrica(config)
    inicio = perf_counter()
    modelo.fit(x_tr, y_tr)
    segundos_ajuste = perf_counter() - inicio
    estado, elegivel = status_convergencia(modelo, config.modelo)
    inicio = perf_counter()
    previsto = modelo.predict(x_val)
    segundos_previsao = perf_counter() - inicio
    metricas = calcular_metricas(y[validacao], previsto)
    treino_previsto = modelo.predict(x_treino_original)
    return {
        "configuracao": asdict(config), "status_convergencia": estado,
        "elegivel": elegivel, "metricas": metricas,
        "f1_macro_ajuste": calcular_metricas(y_treino_original, treino_previsto)["f1_macro"],
        "linhas_ajuste": int(len(y_tr)), "linhas_validacao": int(len(validacao)),
        "colunas": int(x_tr.shape[1]), "indices_validacao": validacao.tolist(),
        "previsoes": np.asarray(previsto).tolist(),
        "tempos_s": {"preparo": segundos_preparo, "ajuste": segundos_ajuste,
                     "previsao": segundos_previsao},
    }


def _versoes():
    resultado = {}
    for pacote in PACOTES:
        try:
            resultado[pacote] = version(pacote)
        except PackageNotFoundError:
            resultado[pacote] = None
    return resultado


def _commit():
    resposta = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent,
                              capture_output=True, text=True)
    return resposta.stdout.strip() if resposta.returncode == 0 else None


def preparar_entrada(entrada, tipo_dados):
    if tipo_dados == "official_training":
        if not isinstance(entrada, (str, os.PathLike)):
            raise ValueError("A origem oficial deve ser o caminho de treino.csv")
        caminho = Path(entrada)
        if caminho.name != "treino.csv":
            raise ValueError("A origem oficial deve ser treino.csv")
        dados = carregar_treino(caminho)
        digest = sha256(caminho.read_bytes()).hexdigest()
        origem = str(caminho.resolve())
    elif tipo_dados == "synthetic_test" and isinstance(entrada, pd.DataFrame):
        dados = entrada.copy(deep=True)
        digest = sha256(pd.util.hash_pandas_object(dados, index=True).values.tobytes()).hexdigest()
        origem = "fixture_sintetica"
    else:
        raise ValueError("Informe treino.csv oficial ou DataFrame sintetico e tipo correspondente")
    if dados.empty or "class" not in dados or dados["class"].isna().any() or not set(dados["class"]).issubset(CLASSES):
        raise ValueError("Alvo class invalido")
    return dados, digest, origem


def resultados_compativeis(primeiro, segundo):
    """Somente avaliacoes do mesmo CSV e dos mesmos indices podem ser pareadas."""
    chaves = ("tipo_dados", "sha256_dados", "protocolo_sha256", "fold_externo",
              "indices_validacao_sha256")
    return all(chave in primeiro and chave in segundo and primeiro[chave] is not None
               and primeiro[chave] == segundo[chave] for chave in chaves)


def executar_avaliacao(entrada, configuracoes, destino, *, tipo_dados,
                      externas=5, internas=3, semente=SEMENTE, fabrica=criar_modelo):
    """Seleciona por folds internos e avalia cada escolha no fold externo.

    O chamador deve guardar destino fora do Git (por exemplo temp/stage3_runs).
    Esta funcao nao e chamada automaticamente nesta fase.
    """
    dados, digest, origem = preparar_entrada(entrada, tipo_dados)
    x, y, grupos, plano = planejar_divisoes(dados, externas=externas, internas=internas,
                                            semente=semente)
    nomes = [c.nome for c in configuracoes]
    if not nomes or len(nomes) != len(set(nomes)):
        raise ValueError("Forneca configuracoes com nomes unicos")
    identificadores = [
        {"fold": numero, "ajuste": treino.tolist(), "validacao": validacao.tolist(),
         "internas": [{"ajuste": a.tolist(), "validacao": v.tolist()} for a, v in interno]}
        for numero, treino, validacao, interno in plano
    ]
    protocolo_sha = sha256(json.dumps(identificadores, sort_keys=True).encode()).hexdigest()
    destino = Path(destino)
    destino.mkdir(parents=True, exist_ok=False)
    manifesto = {
        "tipo_dados": tipo_dados, "origem": origem, "sha256_dados": digest,
        "protocolo_sha256": protocolo_sha, "metodo": "StratifiedGroupKFold_aninhado",
        "papel_quinta_historica": "avaliacao_interna_reutilizada; nenhum fold e teste intocado",
        "semente": semente, "externas": externas, "internas": internas,
        "classes": CLASSES, "configuracoes": [asdict(c) for c in configuracoes],
        "divisoes": identificadores, "versoes": _versoes(),
        "python": sys.version, "plataforma": platform.platform(), "commit": _commit(),
    }
    (destino / "manifesto.json").write_text(json.dumps(manifesto, ensure_ascii=False, indent=2), encoding="utf-8")
    linhas = []
    for numero, treino, validacao, interno in plano:
        desempenho = []
        for config in configuracoes:
            tentativas = []
            for indice, (ajuste, val_interna) in enumerate(interno, 1):
                resultado = avaliar_divisao(x, y, ajuste, val_interna, config, fabrica=fabrica)
                resultado.update(fold_externo=numero, fold_interno=indice, etapa="selecao_interna")
                linhas.append(resultado)
                tentativas.append(resultado)
            if all(r["elegivel"] for r in tentativas):
                desempenho.append((float(np.mean([r["metricas"]["f1_macro"] for r in tentativas])), config))
        if desempenho:
            _, escolhida = max(desempenho, key=lambda par: par[0])
            resultado = avaliar_divisao(x, y, treino, validacao, escolhida, fabrica=fabrica)
            resultado.update(fold_externo=numero, fold_interno=None, etapa="avaliacao_externa",
                             tipo_dados=tipo_dados, sha256_dados=digest,
                             protocolo_sha256=protocolo_sha,
                             indices_validacao_sha256=_digest_indices(validacao))
            linhas.append(resultado)
        else:
            linhas.append({"fold_externo": numero, "etapa": "avaliacao_externa",
                          "status": "sem_configuracao_elegivel", "tipo_dados": tipo_dados,
                          "sha256_dados": digest, "protocolo_sha256": protocolo_sha,
                          "indices_validacao_sha256": _digest_indices(validacao)})
    with (destino / "resultados.jsonl").open("w", encoding="utf-8") as arquivo:
        for linha in linhas:
            arquivo.write(json.dumps(linha, ensure_ascii=False) + "\n")
    return manifesto, linhas
