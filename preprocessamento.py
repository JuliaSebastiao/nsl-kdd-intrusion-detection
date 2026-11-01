"""Funções de diagnóstico, sem aplicar pré-processamento, para a Etapa 1."""

from pathlib import Path

import numpy as np
import pandas as pd

ALVO = "class"
COLUNAS_SUSPEITAS_LEAKAGE = ("tipo_alerta_soc", "score_ids_legado", "tempo_resposta_analista_min")


def carregar_treino(caminho: str | Path) -> pd.DataFrame:
    """Lê o CSV oficial e valida a coluna alvo."""
    caminho = Path(caminho)
    if not caminho.is_file():
        raise FileNotFoundError(f"Base oficial não encontrada: {caminho}")
    dados = pd.read_csv(caminho)
    if ALVO not in dados or dados.empty:
        raise ValueError(f"A base deve ser não vazia e conter {ALVO!r}.")
    return dados.copy()


def identificar_colunas_por_tipo(dados: pd.DataFrame, alvo: str = ALVO) -> tuple[list[str], list[str]]:
    entradas = dados.drop(columns=[alvo], errors="ignore")
    numericas = entradas.select_dtypes(include="number").columns.tolist()
    return numericas, [coluna for coluna in entradas if coluna not in numericas]


def identificar_taxas(dados: pd.DataFrame) -> list[str]:
    """Retorna apenas frações ``*_rate``; ``*_pct`` usa escala percentual."""
    return [c for c in dados.select_dtypes(include="number") if c.lower().endswith("rate")]


def _linha_regra(dados: pd.DataFrame, atributo: str, regra: str, mascara: pd.Series) -> dict:
    valores = dados[atributo]
    return {"atributo": atributo, "regra": regra, "violacoes": int(mascara.sum()),
            "percentual_base": round(100 * mascara.mean(), 3),
            "minimo_observado": valores.min(skipna=True), "maximo_observado": valores.max(skipna=True)}


def analisar_violacoes_dominio(dados: pd.DataFrame) -> pd.DataFrame:
    """Resume regras objetivas; ausências não contam como violações."""
    linhas = []
    for coluna in ("duration", "src_bytes", "dst_bytes"):
        if coluna in dados:
            linhas.append(_linha_regra(dados, coluna, f"{coluna} >= 0", dados[coluna].lt(0).fillna(False)))
    for coluna in identificar_taxas(dados):
        linhas.append(_linha_regra(dados, coluna, "0 <= taxa <= 1",
                                  (dados[coluna].lt(0) | dados[coluna].gt(1)).fillna(False)))
    for coluna, limite in {"count": 511, "srv_count": 511, "dst_host_count": 255}.items():
        if coluna in dados:
            linhas.append(_linha_regra(dados, coluna, f"{coluna} <= {limite}", dados[coluna].gt(limite).fillna(False)))
    if "su_attempted" in dados:
        mascara = (~dados["su_attempted"].isin([0, 1])) & dados["su_attempted"].notna()
        linhas.append(_linha_regra(dados, "su_attempted", "su_attempted em {0, 1}", mascara))
    return pd.DataFrame(linhas)


def resumir_taxas_invalidas(dados: pd.DataFrame) -> dict:
    """Conta linhas únicas com ao menos uma fração fora de [0, 1]."""
    taxas = identificar_taxas(dados)
    mascara = (dados[taxas].lt(0) | dados[taxas].gt(1)).any(axis=1) if taxas else pd.Series(False, index=dados.index)
    return {"registros_com_taxa_invalida": int(mascara.sum()),
            "percentual_base": round(100 * mascara.mean(), 3)}


def exemplos_violacoes(dados: pd.DataFrame, limite: int = 5) -> pd.DataFrame:
    """Retorna até ``limite`` exemplos de cada regra violada."""
    partes = []
    for _, regra in analisar_violacoes_dominio(dados).iterrows():
        coluna, texto = regra["atributo"], regra["regra"]
        if texto.endswith(">= 0"):
            mascara = dados[coluna].lt(0)
        elif texto == "0 <= taxa <= 1":
            mascara = dados[coluna].lt(0) | dados[coluna].gt(1)
        elif texto.startswith("su_attempted"):
            mascara = (~dados[coluna].isin([0, 1])) & dados[coluna].notna()
        else:
            mascara = dados[coluna].gt(float(texto.rsplit(" ", 1)[1]))
        exemplo = dados.loc[mascara, [coluna, ALVO]].head(limite).copy()
        exemplo.insert(0, "regra", texto)
        partes.append(exemplo)
    return pd.concat(partes, ignore_index=True, sort=False) if partes else pd.DataFrame()


def investigar_su_attempted(dados: pd.DataFrame) -> dict[str, pd.DataFrame | int]:
    """Cria as tabelas de auditoria de ``su_attempted`` sem recodificá-lo."""
    if "su_attempted" not in dados:
        raise ValueError("Coluna 'su_attempted' não encontrada.")
    frequencias = dados["su_attempted"].value_counts(dropna=False).rename("frequencia").to_frame()
    frequencias["percentual"] = (100 * frequencias["frequencia"] / len(dados)).round(3)
    subconjunto = dados.loc[dados["su_attempted"].eq(2)].copy()
    entradas = [c for c in dados if c != ALVO]
    return {"frequencias": frequencias,
            "por_classe": pd.crosstab(dados["su_attempted"], dados[ALVO], dropna=False),
            "por_servico": pd.crosstab(dados["su_attempted"], dados.get("service"), dropna=False),
            "por_root_shell": pd.crosstab(dados["su_attempted"], dados.get("root_shell"), dropna=False),
            "registros_com_valor_2": subconjunto,
            "duplicatas_valor_2": int(subconjunto.duplicated(subset=entradas).sum())}


def encontrar_inconsistencias_caixa(dados: pd.DataFrame, colunas: list[str]) -> pd.DataFrame:
    """Encontra categorias que diferem apenas por caixa ou espaços laterais."""
    achados = []
    for coluna in colunas:
        original = dados[coluna].dropna().astype(str)
        normalizado = original.str.strip().str.casefold()
        for chave, valores in original.groupby(normalizado):
            variantes = sorted(valores.unique())
            if len(variantes) > 1:
                achados.append({"atributo": coluna, "valor_normalizado": chave,
                                "variantes_encontradas": ", ".join(variantes),
                                "quantidade_variantes": len(variantes)})
    return pd.DataFrame(achados)


def analisar_duplicatas(dados: pd.DataFrame, alvo: str = ALVO) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Resume duplicatas e retorna grupos de entradas com classes conflitantes."""
    entradas = [c for c in dados if c != alvo]
    grupos = dados.groupby(entradas, dropna=False)[alvo].nunique().rename("classes_distintas")
    chaves = grupos[grupos.gt(1)].reset_index()
    conflitos = dados.merge(chaves, on=entradas, how="inner") if len(chaves) else dados.iloc[0:0].copy()
    if "classes_distintas" not in conflitos:
        conflitos["classes_distintas"] = pd.Series(dtype="int64")
    exatas = int(dados.duplicated().sum())
    mesmas_entradas_mesma_classe = int(dados.duplicated(subset=entradas + [alvo]).sum())
    resumo = pd.DataFrame([
        {"tipo": "duplicatas exatas (inclui class)", "quantidade": exatas},
        {"tipo": "repetições adicionais: mesmas entradas e mesma class", "quantidade": mesmas_entradas_mesma_classe - exatas},
        {"tipo": "registros em grupos com classes diferentes", "quantidade": int(len(conflitos))},
    ])
    resumo["percentual"] = (100 * resumo["quantidade"] / len(dados)).round(3)
    return resumo, conflitos


def verificar_redundancias(dados: pd.DataFrame) -> pd.DataFrame:
    """Testa relações determinísticas suspeitas; não remove nenhuma coluna."""
    testes = [
        ("src_bytes_kb", "src_bytes / 1024", ["src_bytes"], lambda d: d.src_bytes / 1024),
        ("dst_bytes_kb", "dst_bytes / 1024", ["dst_bytes"], lambda d: d.dst_bytes / 1024),
        ("total_bytes", "src_bytes + dst_bytes", ["src_bytes", "dst_bytes"], lambda d: d.src_bytes + d.dst_bytes),
        ("same_srv_pct", "same_srv_rate * 100", ["same_srv_rate"], lambda d: d.same_srv_rate * 100),
    ]
    linhas = []
    for coluna, relacao, dependencias, calculo in testes:
        if coluna not in dados or not set(dependencias).issubset(dados):
            continue
        esperado = calculo(dados)
        comparaveis = dados[coluna].notna() & esperado.notna()
        iguais = np.isclose(dados.loc[comparaveis, coluna], esperado.loc[comparaveis], rtol=0, atol=1e-10)
        linhas.append({"atributo": coluna, "relacao_testada": relacao,
                       "linhas_comparaveis": int(comparaveis.sum()), "correspondencias": int(iguais.sum()),
                       "percentual_correspondencia": round(100 * iguais.mean(), 3) if comparaveis.any() else np.nan,
                       "redundancia_comprovada": bool(comparaveis.any() and iguais.all())})
    if {"protocolo_cod", "protocol_type"}.issubset(dados):
        pares = dados[["protocol_type", "protocolo_cod"]].dropna()
        deterministica = pares.groupby("protocol_type").protocolo_cod.nunique().le(1).all()
        linhas.append({"atributo": "protocolo_cod", "relacao_testada": "codificação determinística de protocol_type",
                       "linhas_comparaveis": len(pares), "correspondencias": len(pares) if deterministica else 0,
                       "percentual_correspondencia": 100.0 if deterministica else 0.0,
                       "redundancia_comprovada": bool(deterministica)})
    return pd.DataFrame(linhas)
