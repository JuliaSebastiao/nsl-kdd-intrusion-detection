"""Plano limitado, verificacao da base e consolidacao reproduzivel da Etapa 3."""

import argparse
from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd

from stage3_evaluation import (CLASSES, CONFIGURACOES_REFERENCIA, Configuracao,
                               calcular_metricas, executar_avaliacao, preparar_entrada)


# Cabecalho observado no treino fornecido, sem inferir atributos da fixture.
COLUNAS_OFICIAIS = """duration protocol_type service flag src_bytes dst_bytes land
wrong_fragment urgent hot num_failed_logins logged_in num_compromised root_shell
su_attempted num_root num_file_creations num_shells num_access_files num_outbound_cmds
is_host_login is_guest_login count srv_count serror_rate srv_serror_rate rerror_rate
srv_rerror_rate same_srv_rate diff_srv_rate srv_diff_host_rate dst_host_count
dst_host_srv_count dst_host_same_srv_rate dst_host_diff_srv_rate dst_host_same_src_port_rate
dst_host_srv_diff_host_rate dst_host_serror_rate dst_host_srv_serror_rate dst_host_rerror_rate
dst_host_srv_rerror_rate src_bytes_kb dst_bytes_kb total_bytes same_srv_pct protocolo_cod
score_ids_legado tipo_alerta_soc tempo_resposta_analista_min class""".split()


def plano_experimentos():
    """15 candidatos, sem produto cartesiano; 5 folds externos x 2 internos."""
    ponderada = CONFIGURACOES_REFERENCIA[0]
    base = replace(ponderada, nome="rf_livre", parametros={**ponderada.parametros, "class_weight": None})
    configs = [base, ponderada,
               replace(base, nome="rf_12", parametros={**base.parametros, "max_depth": 12}),
               replace(ponderada, nome="rf_12_ponderada",
                       parametros={**ponderada.parametros, "max_depth": 12}),
               *CONFIGURACOES_REFERENCIA[1:5]]
    variacoes = [
        replace(base, nome="rf_over", balanceamento="over_0_5"),
        replace(base, nome="rf_under", balanceamento="under_0_5"),
        replace(base, nome="rf_ordinal", codificacao="inteiros"),
        replace(base, nome="rf_sem_escala", escala="nenhuma"),
        replace(base, nome="rf_todas_linhas", linhas="todas"),
        replace(base, nome="rf_redundancias_A", representacao="A"),
        replace(base, nome="rf_pca60", reducao="pca60"),
    ]
    configs.extend(variacoes)
    comparacoes = {
        "Random Forest": [c.nome for c in configs[:4]],
        "SVM": [c.nome for c in configs[4:6]],
        "MLP pequena (120 epocas)": [configs[6].nome],
        "MLP profunda (120 epocas)": [configs[7].nome],
        "preparo:referencia": [base.nome],
        **{f"preparo:{c.nome}": [c.nome] for c in variacoes},
        "selecao_global_aninhada": [c.nome for c in configs],
    }
    return configs, comparacoes


def verificar_base(caminho):
    dados, digest, _ = preparar_entrada(caminho, "official_training")
    if set(dados) != set(COLUNAS_OFICIAIS) or len(dados.columns) != len(COLUNAS_OFICIAIS):
        raise ValueError(f"Esquema diferente: faltam {set(COLUNAS_OFICIAIS)-set(dados)}; "
                         f"extras {set(dados)-set(COLUNAS_OFICIAIS)}")
    categoricas = {"protocol_type", "service", "flag", "tipo_alerta_soc", "class"}
    invalidas = [c for c in dados if c not in categoricas and not pd.api.types.is_numeric_dtype(dados[c])]
    if invalidas:
        raise ValueError(f"Colunas numericas com tipos inesperados: {invalidas}")
    return {
        "sha256": digest, "registros": len(dados), "atributos": len(dados.columns)-1,
        "classes": {k: int(v) for k, v in dados["class"].value_counts().items()},
        "tipos": dados.dtypes.astype(str).to_dict(),
        "ausencias": {k: int(v) for k, v in dados.isna().sum().items()},
        "proveniencia": "Arquivo local identificado pelo usuario/projeto; sem hash externo autenticado",
    }


def _linha_tabela(r, comparacao):
    m = r["metricas"]
    matriz = np.asarray(m["matriz"])
    linha = {"comparacao": comparacao, "fold": r["fold_externo"],
             "configuracao": r["configuracao"]["nome"], "elegivel": r["elegivel"],
             "status_convergencia": r["status_convergencia"],
             **{k: m[k] for k in ("f1_macro", "acuracia", "precisao_macro", "recall_macro")},
             "f1_ajuste": r["f1_macro_ajuste"],
             "gap_f1": r["f1_macro_ajuste"] - m["f1_macro"],
             "segundos_ajuste": r["tempos_s"]["ajuste"],
             "falsos_positivos": int(matriz[0, 1]), "falsos_negativos": int(matriz[1, 0])}
    for classe in CLASSES:
        for k, v in m["por_classe"][classe].items():
            linha[f"{k}_{classe}"] = v
    return linha


def consolidar(destino):
    """Recalcula metricas; nao permite ranking com folds faltantes ou inelegiveis."""
    destino = Path(destino)
    manifesto = json.loads((destino / "manifesto.json").read_text(encoding="utf-8"))
    registros = [json.loads(l) for l in (destino / "resultados.jsonl").read_text(encoding="utf-8").splitlines()]
    nomes = [c["nome"] for c in manifesto["configuracoes"]]
    esperadas = {(d["fold"], i, n) for d in manifesto["divisoes"]
                 for i in range(1, manifesto["internas"]+1) for n in nomes}
    internas = [r for r in registros if r["etapa"] == "selecao_interna"]
    chaves = [(r["fold_externo"], r["fold_interno"], r["configuracao"]["nome"]) for r in internas]
    if set(chaves) != esperadas or len(chaves) != len(esperadas):
        raise ValueError("Selecao interna incompleta ou duplicada")
    tabelas, matrizes, oof = [], {}, {}
    for r in registros:
        if "metricas" not in r:
            continue
        m = calcular_metricas(r["rotulos_reais"], r["previsoes"])
        if m != r["metricas"]:
            raise ValueError("Metricas nao correspondem as previsoes persistidas")
        d = manifesto["divisoes"][r["fold_externo"]-1]
        val = (d["validacao"] if r["etapa"] == "avaliacao_externa"
               else d["internas"][r["fold_interno"]-1]["validacao"])
        if val != r["indices_validacao"] or len(val) != len(r["previsoes"]):
            raise ValueError("Indices persistidos divergentes do manifesto")
        if r["etapa"] != "avaliacao_externa":
            continue
        for comparacao in r["comparacoes"]:
            tabelas.append(_linha_tabela(r, comparacao))
            matrizes.setdefault(comparacao, np.zeros((2, 2), dtype=int))
            matrizes[comparacao] += np.array(m["matriz"])
            oof.setdefault(comparacao, []).extend(val)
    folds = pd.DataFrame(tabelas)
    if folds.empty:
        raise ValueError("Nenhuma avaliacao externa concluida")
    metricas = ["f1_macro", "acuracia", "precisao_macro", "recall_macro", "gap_f1", "segundos_ajuste",
                *[f"{m}_{c}" for c in CLASSES for m in ("precision", "recall", "f1-score")]]
    resumos = []
    total = sum(len(d["validacao"]) for d in manifesto["divisoes"])
    for comparacao in manifesto["comparacoes"]:
        tabela = folds[folds.comparacao == comparacao]
        completo = (len(tabela) == manifesto["externas"]
                    and tabela.fold.nunique() == manifesto["externas"]
                    and sorted(oof.get(comparacao, [])) == list(range(total)))
        elegivel = bool(completo and tabela.elegivel.all())
        r = {"comparacao": comparacao, "completo": completo, "elegivel": elegivel,
             "divisoes": len(tabela)}
        for m in metricas:
            r[m+"_media"] = float(tabela[m].mean()) if len(tabela) else None
            r[m+"_desvio"] = float(tabela[m].std(ddof=1)) if len(tabela)>1 else None
        r["matriz_oof"] = matrizes.get(comparacao, np.zeros((2,2),dtype=int)).tolist()
        resumos.append(r)
    resumo = pd.DataFrame(resumos)
    # O ranking externo seleciona um pipeline: nao e um teste final independente.
    candidatos = resumo[resumo.elegivel & ~resumo.comparacao.eq("selecao_global_aninhada")]
    ranking = candidatos.sort_values(["f1_macro_media", "recall_anomaly_media", "comparacao"],
                                      ascending=[False, False, True], kind="stable")
    vencedor = None if ranking.empty else str(ranking.iloc[0].comparacao)
    internas_tabela = pd.DataFrame([{
        **_linha_tabela(r, r["configuracao"]["nome"]), "fold_interno": r["fold_interno"]
    } for r in internas])
    # Todos os candidatos, inclusive nao convergidos, ficam visiveis.
    configuracoes = internas_tabela.groupby("configuracao", sort=False).agg(
        f1_macro_media=("f1_macro", "mean"), f1_macro_desvio=("f1_macro", "std"),
        recall_anomaly_media=("recall_anomaly", "mean"), elegivel=("elegivel", "all"),
        ajustes=("fold", "size")).reset_index()
    escolhida = None
    if vencedor:
        ns = manifesto["comparacoes"][vencedor]
        aptas = configuracoes[configuracoes.configuracao.isin(ns) & configuracoes.elegivel]
        aptas = aptas.assign(ordem=aptas.configuracao.map({n:i for i,n in enumerate(nomes)}))
        if len(aptas):
            escolhida = str(aptas.sort_values(["f1_macro_media", "recall_anomaly_media", "ordem"],
                                              ascending=[False,False,True]).iloc[0].configuracao)
    return manifesto, folds, resumo, configuracoes, {
        "comparacao_selecionada": vencedor, "configuracao_recomendada": escolhida,
        "criterio": "F1 macro externo medio; recall anomaly; nome em empate. Configuracao por medias internas.",
        "limite": "Escolha informada pela comparacao externa; desempenho do vencedor nao e estimativa final independente. "
                  "Folds internos sobrepostos nao sao repeticoes independentes. Quinta historica reutilizada.",
        "tipo_dados": manifesto["tipo_dados"],
    }


def gerar_relatorio(destino, saida):
    manifesto, folds, resumo, configuracoes, selecao = consolidar(destino)
    saida = Path(saida)
    saida.mkdir(parents=True, exist_ok=True)
    folds.to_csv(saida / "resultados_por_divisao.csv", index=False)
    resumo.drop(columns="matriz_oof").to_csv(saida / "comparacao_modelos.csv", index=False)
    configuracoes.to_csv(saida / "configuracoes_internas.csv", index=False)
    publico = {k:v for k,v in manifesto.items() if k not in ("origem", "divisoes")}
    publico.update(selecao=selecao, resumos=resumo.to_dict(orient="records"),
                   numero_ajustes_internos=len(manifesto["configuracoes"])*manifesto["externas"]*manifesto["internas"])
    (saida / "resultados.json").write_text(json.dumps(publico, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    familias = resumo[~resumo.comparacao.str.startswith("preparo:") & ~resumo.comparacao.eq("selecao_global_aninhada")]
    titulo = "BASE OFICIAL" if manifesto["tipo_dados"] == "official_training" else "TESTE SINTETICO"

    def salvar(fig, nome):
        fig.suptitle(titulo + " - validacao agrupada no treino", fontsize=10)
        fig.tight_layout()
        fig.savefig(saida / nome, dpi=160, bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.barh(familias.comparacao, familias.f1_macro_media, xerr=familias.f1_macro_desvio, capsize=4)
    ax.set(xlabel="F1 macro (media e desvio amostral de 5 folds)", xlim=(0,1))
    salvar(fig, "f1_algoritmos.png")
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(familias))
    ax.bar(x-.18, familias.precision_anomaly_media, .36, label="Precisao anomaly")
    ax.bar(x+.18, familias.recall_anomaly_media, .36, label="Recall anomaly")
    ax.set_xticks(x, familias.comparacao, rotation=15, ha="right")
    ax.set(ylim=(0,1), ylabel="Media entre folds")
    ax.legend()
    salvar(fig, "precisao_recall.png")
    fig, ax = plt.subplots(figsize=(9,4))
    for nome in familias.comparacao:
        t = folds[folds.comparacao==nome].sort_values("fold")
        ax.plot(t.fold,t.f1_macro,marker="o",label=nome)
    ax.set(xlabel="Divisao externa",ylabel="F1 macro",xticks=range(1,manifesto["externas"]+1))
    ax.legend(fontsize=8)
    salvar(fig, "variacao_divisoes.png")
    fig, axes = plt.subplots(2,2,figsize=(9,7))
    for ax, (_,r) in zip(axes.flat,familias.iterrows()):
        cm=np.asarray(r.matriz_oof)
        ax.imshow(cm,cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j,i,str(cm[i,j]),ha="center",va="center",color="white" if cm[i,j]>cm.max()/2 else "black")
        ax.set(xticks=[0,1],xticklabels=CLASSES,yticks=[0,1],yticklabels=CLASSES,
               xlabel="Previsto",ylabel="Real",title=r.comparacao)
    salvar(fig,"matrizes_confusao.png")
    prep = resumo[resumo.comparacao.str.startswith("preparo:")]
    fig, ax=plt.subplots(figsize=(9,5))
    ax.barh(prep.comparacao.str.replace("preparo:","",regex=False),prep.f1_macro_media,
            xerr=prep.f1_macro_desvio,capsize=4)
    ax.set(xlabel="F1 macro (media e desvio); RF livre sem ponderacao",xlim=(0,1))
    salvar(fig,"efeito_preprocessamento.png")
    return folds, resumo, configuracoes, selecao


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dados", default="data/raw/treino.csv")
    p.add_argument("--destino", default="temp/stage3_runs/comparacao_cpu")
    p.add_argument("--saida", default="temp/stage3_runs/comparacao_cpu/resumo")
    p.add_argument("--somente-relatorio",action="store_true")
    args=p.parse_args()
    if not args.somente_relatorio:
        diagnostico=verificar_base(args.dados)
        print(json.dumps(diagnostico,ensure_ascii=False),flush=True)
        configs,comparacoes=plano_experimentos()
        executar_avaliacao(args.dados,configs,args.destino,tipo_dados="official_training",
                          externas=5,internas=2,comparacoes=comparacoes,retomar=True)
        if sha256(Path(args.dados).read_bytes()).hexdigest()!=diagnostico["sha256"]:
            raise RuntimeError("Arquivo de entrada foi alterado durante a execucao")
        (Path(args.destino)/"base_verificada.json").write_text(json.dumps(diagnostico,indent=2),encoding="utf-8")
    _,resumo,_,selecao=gerar_relatorio(args.destino,args.saida)
    print(resumo[["comparacao","f1_macro_media","f1_macro_desvio","elegivel"]].to_string(index=False))
    print(json.dumps(selecao,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
