"""Prepara??o determin?stica e divis?es agrupadas das etapas 2 e 3."""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold

ALVO = "class"
COLUNAS_EXCLUIDAS_EXPERIMENTOS = (
    "num_outbound_cmds", "tempo_resposta_analista_min", "tipo_alerta_soc", "score_ids_legado",
)
SEMENTE = 42
NOMINAIS = ["protocol_type", "service", "flag", "protocolo_cod"]
COPIAS = ["src_bytes_kb", "dst_bytes_kb", "same_srv_pct", "protocolo_cod"]


def carregar_treino(caminho: str | Path) -> pd.DataFrame:
    """Lê o CSV oficial e valida a coluna alvo."""
    caminho = Path(caminho)
    if not caminho.is_file():
        raise FileNotFoundError(f"Base oficial não encontrada: {caminho}")
    dados = pd.read_csv(caminho)
    if ALVO not in dados or dados.empty:
        raise ValueError(f"A base deve ser não vazia e conter {ALVO!r}.")
    if dados[ALVO].isna().any() or not set(dados[ALVO].unique()).issubset({"normal", "anomaly"}):
        raise ValueError("A coluna class deve conter somente normal ou anomaly, sem ausências.")
    return dados.copy()


def identificar_taxas(dados: pd.DataFrame) -> list[str]:
    """Retorna apenas frações ``*_rate``; ``*_pct`` usa escala percentual."""
    return [c for c in dados.select_dtypes(include="number") if c.lower().endswith("rate")]


def preparar_base(dados):
    entradas = dados.drop(columns=["class", *COLUNAS_EXCLUIDAS_EXPERIMENTOS])
    for coluna in NOMINAIS:
        entradas[coluna] = entradas[coluna].astype("string").str.strip().str.casefold().replace("", pd.NA)

    invalidos = {}
    for coluna in ("duration", "src_bytes", "dst_bytes"):
        invalidos[coluna] = entradas[coluna].lt(0)
    for coluna in identificar_taxas(entradas):
        invalidos[coluna] = entradas[coluna].notna() & ~entradas[coluna].between(0, 1)
    invalidos["src_bytes_kb"] = invalidos["src_bytes"] | entradas["src_bytes_kb"].lt(0)
    invalidos["dst_bytes_kb"] = invalidos["dst_bytes"] | entradas["dst_bytes_kb"].lt(0)
    invalidos["total_bytes"] = invalidos["src_bytes"] | invalidos["dst_bytes"] | entradas["total_bytes"].lt(0)
    percentual = entradas["same_srv_pct"]
    invalidos["same_srv_pct"] = invalidos["same_srv_rate"] | (percentual.notna() & ~percentual.between(0, 100))
    for coluna, mascara in invalidos.items():
        entradas[coluna] = entradas[coluna].mask(mascara)

    representacoes = {
        "A": entradas,
        "B": entradas.drop(columns=COPIAS),
        "C": entradas.drop(columns=[*COPIAS, "total_bytes"]),
    }
    assert [m.shape[1] for m in representacoes.values()] == [45, 41, 40]
    grupos = representacoes["C"].groupby(list(representacoes["C"]), dropna=False, sort=False).ngroup()
    assert grupos.notna().all()
    return representacoes, grupos


def escolher_linhas(x, y, indices, regra):
    if regra == "todas":
        return indices
    chaves = x.iloc[indices].assign(rotulo=y[indices])
    mascara = np.ones(len(indices), dtype=bool)
    if regra in {"sem_conflitos", "sem_ambos"}:
        grupos = chaves.groupby(list(x), dropna=False, sort=False)["rotulo"]
        mascara &= grupos.transform("nunique").to_numpy() == 1
    if regra in {"sem_duplicatas", "sem_ambos"}:
        mascara &= ~chaves.duplicated().to_numpy()
    return indices[mascara]


def separar_divisoes(representacoes, y, grupos):
    divisor = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEMENTE)
    particoes = list(divisor.split(representacoes["C"], y, grupos))
    reserva = particoes[-1][1]
    divisoes = []
    for fold, (treino, validacao) in enumerate(particoes, 1):
        if fold != 5:
            treino = np.setdiff1d(treino, reserva)
            assert set(grupos.iloc[treino]).isdisjoint(grupos.iloc[reserva])
            assert set(grupos.iloc[validacao]).isdisjoint(grupos.iloc[reserva])
        assert set(grupos.iloc[treino]).isdisjoint(grupos.iloc[validacao])
        divisoes.append((fold, treino, validacao))
    return divisoes