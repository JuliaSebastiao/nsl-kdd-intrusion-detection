"""Transforma??es ajustadas somente na parcela de treino de cada divis?o."""

import numpy as np
import pandas as pd
from sklearn import config_context
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder, RobustScaler, StandardScaler
from dados_etapa2 import NOMINAIS, SEMENTE, escolher_linhas


def transformar_numericas(treino, validacao, metodo, escala):
    treino = treino.to_numpy(dtype=np.float64, na_value=np.nan)
    validacao = validacao.to_numpy(dtype=np.float64, na_value=np.nan)
    # Uma entrada sem observa??es no ajuste recebe zero em ambas as parcelas.
    # Valores observados apenas na valida??o n?o participam do ajuste do KNN.
    if metodo == "knn":
        vazias = np.isnan(treino).all(axis=0)
        treino[:, vazias] = 0.0
        validacao[:, vazias] = 0.0
    if metodo == "mediana_indicador":
        tem_ausencia = np.isnan(treino).any(axis=0)
        indicadores_treino = np.isnan(treino[:, tem_ausencia])
        indicadores_val = np.isnan(validacao[:, tem_ausencia])
    if metodo == "knn":
        # Limita o custo de vizinhança a uma amostra sem rótulos da parte de treino.
        normalizador = StandardScaler().fit(treino)
        amostra = np.random.default_rng(SEMENTE).choice(len(treino), size=min(2000, len(treino)), replace=False)
        imputador = KNNImputer(n_neighbors=5, keep_empty_features=True)
        imputador.fit(normalizador.transform(treino[amostra]))
        with config_context(working_memory=48):
            treino = normalizador.inverse_transform(imputador.transform(normalizador.transform(treino)))
            validacao = normalizador.inverse_transform(imputador.transform(normalizador.transform(validacao)))
    else:
        estrategia = "mean" if metodo == "media" else "median"
        imputador = SimpleImputer(strategy=estrategia, keep_empty_features=True)
        treino, validacao = imputador.fit_transform(treino), imputador.transform(validacao)
    normalizadores = {"zscore": StandardScaler, "robusta": RobustScaler, "minmax": MinMaxScaler}
    if escala in normalizadores:
        normalizador = normalizadores[escala]()
        treino, validacao = normalizador.fit_transform(treino), normalizador.transform(validacao)
    if metodo == "mediana_indicador":
        treino = np.column_stack((treino, indicadores_treino))
        validacao = np.column_stack((validacao, indicadores_val))
    return treino, validacao


def transformar_categoricas(treino, validacao, metodo, codificacao):
    estrategia = "most_frequent" if metodo == "moda" else "constant"
    colunas = list(treino)
    colunas_treino, colunas_validacao = [], []
    for coluna in colunas:
        valores_treino = treino[[coluna]].to_numpy(dtype=object, na_value=np.nan)
        valores_validacao = validacao[[coluna]].to_numpy(dtype=object, na_value=np.nan)
        # A moda n?o existe quando toda a coluna do ajuste est? ausente.
        # Nesse caso a categoria expl?cita mant?m a posi??o da coluna.
        estrategia_coluna = "constant" if pd.isna(valores_treino).all() else estrategia
        imputador = SimpleImputer(strategy=estrategia_coluna, fill_value="__ausente__")
        colunas_treino.append(imputador.fit_transform(valores_treino))
        colunas_validacao.append(imputador.transform(valores_validacao))
    treino = np.column_stack(colunas_treino)
    validacao = np.column_stack(colunas_validacao)
    if codificacao == "inteiros":
        codificador = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        return codificador.fit_transform(treino), codificador.transform(validacao)
    cod_treino, cod_val = [], []
    for i, coluna in enumerate(colunas):
        agrupar = codificacao == "onehot_raros" and coluna == "service"
        codificador = OneHotEncoder(
            sparse_output=False,
            min_frequency=20 if agrupar else None,
            handle_unknown="infrequent_if_exist" if agrupar else "ignore",
        )
        cod_treino.append(codificador.fit_transform(treino[:, i:i + 1]))
        cod_val.append(codificador.transform(validacao[:, i:i + 1]))
    return np.column_stack(cod_treino), np.column_stack(cod_val)


def transformar(x, y, indices, validacao, config):
    indices = escolher_linhas(x, y, indices, config.linhas)
    treino, validacao = x.iloc[indices], x.iloc[validacao]
    categoricas = [c for c in NOMINAIS if c in x]
    numericas = x.columns.drop(categoricas).tolist()
    num_treino, num_val = transformar_numericas(treino[numericas], validacao[numericas], config.numericos, config.escala)
    cat_treino, cat_val = transformar_categoricas(treino[categoricas], validacao[categoricas], config.nominais, config.codificacao)
    return np.column_stack((num_treino, cat_treino)), np.column_stack((num_val, cat_val)), y[indices]