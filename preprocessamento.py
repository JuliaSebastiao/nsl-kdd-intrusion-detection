"""Leitura do treino NSL-KDD e comparação das estratégias de conflito.

As funções não leem KDDTest+ e não alteram os dados recebidos.
Na validação, aplicar_estrategia deve receber apenas o treino da rodada.
"""

from pathlib import Path
import io
import zipfile

import pandas as pd


# Os 41 atributos descritivos do esquema original do NSL-KDD.
ATRIBUTOS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
]
CATEGORICOS = ["protocol_type", "service", "flag"]

# Decisão aprovada em 14/10/2026: num_outbound_cmds é zero em todo o treino.
# O esquema original continua com 41 atributos; os modelos recebem estes 40.
ATRIBUTOS_MODELO = [c for c in ATRIBUTOS if c != "num_outbound_cmds"]

# Fonte: https://kdd.ics.uci.edu/databases/kddcup99/training_attack_types
FAMILIAS = {
    "normal": "Normal", "back": "DoS", "buffer_overflow": "U2R",
    "ftp_write": "R2L", "guess_passwd": "R2L", "imap": "R2L",
    "ipsweep": "Probe", "land": "DoS", "loadmodule": "U2R", "multihop": "R2L",
    "neptune": "DoS", "nmap": "Probe", "perl": "U2R", "phf": "R2L",
    "pod": "DoS", "portsweep": "Probe", "rootkit": "U2R", "satan": "Probe",
    "smurf": "DoS", "spy": "R2L", "teardrop": "DoS",
    "warezclient": "R2L", "warezmaster": "R2L",
}
MODOS_ALVO = ("original", "familias", "binario")
ESTRATEGIAS = ("manter", "remover_conflitos")


def carregar_treino(caminho_zip):
    """Lê somente KDDTrain+.txt; mantém a linha original como índice (base 1)."""
    caminho_zip = Path(caminho_zip)
    with zipfile.ZipFile(caminho_zip) as arquivo:
        # Não listar, extrair ou ler os arquivos de teste.
        conteudo = arquivo.read("KDDTrain+.txt")
    dados = pd.read_csv(io.BytesIO(conteudo), header=None, dtype=str, keep_default_na=False)
    if dados.shape[1] != 43:
        raise ValueError(f"Esperadas 43 colunas; encontradas {dados.shape[1]}.")
    dados.columns = ATRIBUTOS + ["attack_type", "difficulty"]
    for coluna in [c for c in ATRIBUTOS if c not in CATEGORICOS] + ["difficulty"]:
        # Converter a representação para números não é normalizar ou imputar.
        # Um valor ilegível gera erro para investigação, em vez de correção silenciosa.
        dados[coluna] = pd.to_numeric(dados[coluna], errors="raise")
    dados.index = pd.RangeIndex(1, len(dados) + 1, name="linha_original")
    return dados


def selecionar_entradas(dados):
    """Prepara uma cópia dos 40 atributos aprovados para os futuros modelos.

    A mesma seleção é aplicada ao treino e aos conjuntos de avaliação,
    preservando linhas e índices. A base original e o diagnóstico mantêm
    num_outbound_cmds. Não selecionar outras constantes automaticamente.
    """
    return dados.loc[:, ATRIBUTOS_MODELO].copy()


def construir_alvo(dados, modo):
    """Constrói uma cópia do alvo para diagnóstico, sem modificar attack_type."""
    if modo not in MODOS_ALVO:
        raise ValueError(f"Modo desconhecido: {modo!r}. Escolha entre {MODOS_ALVO}.")
    original = dados["attack_type"]
    if original.isna().any():
        raise ValueError("O alvo contém valores ausentes.")
    if modo == "original":
        return original.copy().rename("classe")
    desconhecidos = sorted(set(original) - set(FAMILIAS))
    if desconhecidos:
        raise ValueError(f"Rótulos sem mapeamento de família: {desconhecidos}.")
    if modo == "familias":
        return original.map(FAMILIAS).rename("classe")
    return original.map(lambda valor: "Normal" if valor == "normal" else "Ataque").rename("classe")


def _validar_entradas(X, y):
    if not isinstance(X, pd.DataFrame) or not isinstance(y, pd.Series):
        raise TypeError("X deve ser um DataFrame e y uma Series.")
    if not X.index.is_unique or not y.index.is_unique or not X.index.equals(y.index):
        raise ValueError("X e y precisam de índices únicos, iguais e na mesma ordem.")
    if not len(X.columns) or not X.columns.is_unique:
        raise ValueError("X deve conter atributos com nomes únicos.")
    proibidas = {"attack_type", "difficulty", "classe", "_rotulo"} & set(X.columns)
    if proibidas:
        raise ValueError(f"Rótulo ou metadado incluído na chave de atributos: {sorted(proibidas)}.")
    if y.isna().any():
        raise ValueError("O alvo contém valores ausentes.")


def identificar_conflitos(X, y):
    """Marca todos os registros com atributos iguais e mais de uma classe.

    A definição de conflito depende do y fornecido. Repetições com a mesma
    classe permanecem. A dificuldade e o rótulo nunca entram na chave.
    """
    _validar_entradas(X, y)
    if X.empty:
        return pd.Series(False, index=X.index, dtype=bool, name="conflito")
    tabela = X.copy()
    tabela["_rotulo"] = y
    diferentes = tabela.groupby(
        list(X.columns), dropna=False, sort=False
    )["_rotulo"].transform("nunique")
    return diferentes.gt(1).rename("conflito")


def aplicar_estrategia(X_treino, y_treino, estrategia):
    """Retorna X, y e índices removidos, sem alterar as entradas.

    Deve receber somente os dados de treinamento da rodada. Validação e
    teste não são argumentos desta função nem devem passar por essa limpeza.
    """
    _validar_entradas(X_treino, y_treino)
    if estrategia not in ESTRATEGIAS:
        raise ValueError(f"Estratégia desconhecida: {estrategia!r}.")
    if estrategia == "manter":
        remover = pd.Series(False, index=X_treino.index, dtype=bool)
    else:
        remover = identificar_conflitos(X_treino, y_treino)
    return (
        X_treino.loc[~remover].copy(),
        y_treino.loc[~remover].copy(),
        X_treino.index[remover].copy(),
    )


def impacto_por_classe(y, remover):
    """Conta perdas absolutas e relativas, sem descartar ou alterar registros."""
    if not y.index.equals(remover.index) or remover.dtype != bool:
        raise ValueError("A máscara booleana precisa estar alinhada com o alvo.")
    resumo = pd.concat(
        [y.value_counts().rename("original"), y.loc[remover].value_counts().rename("removidos")],
        axis=1,
    ).fillna(0).astype(int)
    resumo["restantes"] = resumo["original"] - resumo["removidos"]
    resumo["percentual_removido"] = (100 * resumo["removidos"] / resumo["original"]).round(2)
    return resumo.sort_values(["percentual_removido", "original"], ascending=[False, False])
