"""Fixtures pequenas; nenhuma metrica destes testes representa a base oficial."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from dados_etapa2 import preparar_base
from preparar_notebooks_etapa3 import NOTEBOOKS, sincronizar
from stage3_evaluation import (
    Configuracao, avaliar_divisao, calcular_metricas, criar_modelo,
    executar_avaliacao, planejar_divisoes, preparar_entrada, resultados_compativeis,
    status_convergencia,
)
from tecnicas_etapa2 import transformar_categoricas, transformar_numericas


def fixture():
    """40 entradas em C, com 50 grupos de duas linhas e 20% de anomalias."""
    grupo = np.repeat(np.arange(50), 2)
    n = len(grupo)
    dados = {
        "protocol_type": np.where(grupo % 2, "tcp", "udp"),
        "service": np.where(grupo % 3, "http", "ftp"),
        "flag": np.where(grupo % 2, "SF", "S0"),
        "duration": grupo.astype(float),
        "src_bytes": grupo * 1024.0,
        "dst_bytes": grupo * 2048.0,
        "same_srv_rate": (grupo % 10) / 10,
    }
    dados.update({f"audit_feature_{i}": (grupo + i) % 17 for i in range(33)})
    dados.update({
        "num_outbound_cmds": np.zeros(n),
        "tempo_resposta_analista_min": np.ones(n),
        "tipo_alerta_soc": ["OK"] * n,
        "score_ids_legado": np.zeros(n),
        "src_bytes_kb": grupo.astype(float),
        "dst_bytes_kb": grupo * 2.0,
        "same_srv_pct": (grupo % 10) * 10.0,
        "protocolo_cod": (grupo % 2).astype(float),
        "total_bytes": grupo * 3072.0,
        "class": np.where(grupo % 5 == 0, "anomaly", "normal"),
    })
    return pd.DataFrame(dados)


class PreparoTest(unittest.TestCase):
    def test_moda_preserva_coluna_vazia_e_categoria_desconhecida(self):
        treino = pd.DataFrame({"service": [None, None, None],
                               "flag": ["sf", "s0", "sf"]})
        validacao = pd.DataFrame({"service": ["novo", None],
                                  "flag": ["sf", "s0"]})
        original = validacao.copy(deep=True)
        for codificacao in ("onehot", "onehot_raros", "inteiros"):
            x_tr, x_val = transformar_categoricas(treino, validacao, "moda", codificacao)
            self.assertEqual(x_tr.shape[0], 3)
            self.assertEqual(x_val.shape[0], 2)
            self.assertEqual(x_tr.shape[1], x_val.shape[1])
            self.assertTrue(np.isfinite(x_val).all())
            if codificacao == "inteiros":
                self.assertEqual(x_val[0, 0], -1)
                self.assertEqual(x_val[1, 0], 0)
            elif codificacao == "onehot":
                self.assertEqual(x_val[0, 0], 0)
                self.assertEqual(x_val[1, 0], 1)
            else:
                # Com infrequent_if_exist, a categoria nova pode usar o grupo raro.
                self.assertIn(x_val[0, 0], (0, 1))
        assert_frame_equal(validacao, original)

    def test_moda_regular_mantem_comportamento(self):
        treino = pd.DataFrame({"service": ["http", "http", None]})
        validacao = pd.DataFrame({"service": [None, "novo"]})
        x_tr, x_val = transformar_categoricas(treino, validacao, "moda", "onehot")
        self.assertEqual(x_tr.shape, (3, 1))
        self.assertEqual(x_val[:, 0].tolist(), [1.0, 0.0])

    def test_knn_coluna_vazia_nao_usa_validacao(self):
        treino = pd.DataFrame({"vazia": [np.nan] * 5,
                               "observada": [1., 2., np.inf, 4., 5.]})
        validacao = pd.DataFrame({"vazia": [1000., np.nan, np.inf],
                                  "observada": [2., np.nan, np.inf]})
        original = validacao.copy(deep=True)
        for escala in ("zscore", "minmax", "robusta", "nenhuma"):
            x_tr, x_val = transformar_numericas(treino, validacao, "knn", escala)
            self.assertEqual(x_tr.shape, (5, 2))
            self.assertEqual(x_val.shape, (3, 2))
            self.assertTrue(np.isfinite(x_tr).all())
            self.assertTrue(np.isfinite(x_val).all())
            self.assertTrue((x_tr[:, 0] == 0).all())
            self.assertTrue((x_val[:, 0] == 0).all())
        assert_frame_equal(validacao, original)

    def test_estatisticas_ajustadas_apenas_no_treino(self):
        treino = pd.DataFrame({"valor": [0., 2., np.nan]})
        val_a = pd.DataFrame({"valor": [1000.]})
        val_b = pd.DataFrame({"valor": [1000000.]})
        tr_a, _ = transformar_numericas(treino, val_a, "mediana", "zscore")
        tr_b, _ = transformar_numericas(treino, val_b, "mediana", "zscore")
        np.testing.assert_array_equal(tr_a, tr_b)
        self.assertAlmostEqual(tr_a[2, 0], 0)

    def test_copias_independentes_sincronizadas(self):
        for nome in NOTEBOOKS:
            with self.subTest(notebook=nome):
                self.assertTrue(sincronizar(nome, check=True))


class ProtocoloTest(unittest.TestCase):
    def test_grupos_isolados_repetiveis_e_validacao_preservada(self):
        dados = fixture()
        original = dados.copy(deep=True)
        x, y, grupos, plano = planejar_divisoes(dados, internas=2)
        _, _, _, repetido = planejar_divisoes(dados, internas=2)
        assert_frame_equal(dados, original)
        self.assertEqual(x.shape[1], 40)
        self.assertEqual(len(plano), 5)
        np.testing.assert_array_equal(np.sort(np.concatenate([p[2] for p in plano])),
                                      np.arange(len(dados)))
        for (_, treino, val, internos), (_, tr2, va2, int2) in zip(plano, repetido):
            np.testing.assert_array_equal(treino, tr2)
            np.testing.assert_array_equal(val, va2)
            self.assertFalse(set(grupos.iloc[treino]) & set(grupos.iloc[val]))
            for (ajuste, interna), (a2, v2) in zip(internos, int2):
                np.testing.assert_array_equal(ajuste, a2)
                np.testing.assert_array_equal(interna, v2)
                self.assertFalse(set(grupos.iloc[ajuste]) & set(grupos.iloc[interna]))
                self.assertFalse(set(interna) & set(val))
        self.assertEqual(len(y), len(dados))

    def test_balanceamento_somente_no_ajuste_e_interfaces(self):
        dados = fixture()
        x, y, _, plano = planejar_divisoes(dados, internas=2)
        _, treino, val, _ = plano[0]
        original = x.copy(deep=True)
        base = Configuracao("rf", "random_forest", {"n_estimators": 8,
                             "min_samples_leaf": 2, "class_weight": "balanced_subsample"})
        over = Configuracao("rf_over", "random_forest", {"n_estimators": 8,
                             "min_samples_leaf": 2, "class_weight": "balanced_subsample"},
                             balanceamento="over_0_5")
        sem = avaliar_divisao(x, y, treino, val, base)
        com = avaliar_divisao(x, y, treino, val, over)
        self.assertEqual(sem["linhas_validacao"], com["linhas_validacao"])
        self.assertEqual(sem["indices_validacao"], com["indices_validacao"])
        self.assertGreater(com["linhas_ajuste"], sem["linhas_ajuste"])
        self.assertEqual(sem["colunas"], com["colunas"])
        self.assertEqual(len(com["previsoes"]), len(val))
        self.assertTrue(com["elegivel"])
        assert_frame_equal(x, original)
        for familia in ("svm_linear", "svm_rbf", "mlp_pequena", "mlp_profunda", "pytorch_profunda"):
            parametros = {"epocas": 2} if familia.startswith("mlp_") or familia == "pytorch_profunda" else {}
            modelo = criar_modelo(Configuracao(familia, familia, parametros))
            self.assertTrue(callable(modelo.fit))
            self.assertTrue(callable(modelo.predict))

    def test_metricas_e_ambas_as_classes_na_ordem_fixa(self):
        real = ["normal", "normal", "anomaly", "anomaly"]
        previsto = ["normal", "anomaly", "normal", "anomaly"]
        metricas = calcular_metricas(real, previsto)
        self.assertEqual(metricas["matriz"], [[1, 1], [1, 1]])
        self.assertAlmostEqual(metricas["f1_macro"], 0.5)
        self.assertAlmostEqual(metricas["acuracia"], 0.5)
        self.assertEqual(list(metricas["por_classe"]), ["normal", "anomaly"])
        self.assertEqual(metricas["por_classe"]["anomaly"]["support"], 2)
        somente_normal = calcular_metricas(["normal", "normal"], ["normal", "normal"])
        self.assertEqual(somente_normal["matriz"], [[2, 0], [0, 0]])
        self.assertAlmostEqual(somente_normal["f1_macro"], 0.5)
        real_publicado = ["normal"] * 13095 + ["anomaly"] * 788
        previsao_publicada = (["normal"] * 12926 + ["anomaly"] * 169
                               + ["normal"] * 218 + ["anomaly"] * 570)
        self.assertAlmostEqual(calcular_metricas(real_publicado, previsao_publicada)["f1_macro"],
                               0.865906, places=5)

    def test_nao_convergido_inelegivel_mesmo_com_metricas(self):
        class SVMIncompleta:
            fit_status_ = 1
            def fit(self, x, y):
                return self
            def predict(self, x):
                return np.array(["normal"] * len(x))
        dados = fixture()
        x, y, _, plano = planejar_divisoes(dados, internas=2)
        _, treino, val, _ = plano[0]
        config = Configuracao("nao_convergida", "svm_rbf")
        resultado = avaliar_divisao(x, y, treino, val, config,
                                    fabrica=lambda _: SVMIncompleta())
        self.assertFalse(resultado["elegivel"])
        self.assertEqual(resultado["status_convergencia"], "fit_status_1")
        self.assertIn("f1_macro", resultado["metricas"])
        self.assertEqual(status_convergencia(object(), "svm_rbf"),
                         ("desconhecido", False))

    def test_execucao_sintetica_registra_manifesto_e_divisoes(self):
        dados = fixture()
        config = Configuracao("rf_curta", "random_forest", {"n_estimators": 5})
        with tempfile.TemporaryDirectory() as pasta:
            destino = Path(pasta) / "rodada"
            manifesto, linhas = executar_avaliacao(
                dados, [config], destino, tipo_dados="synthetic_test",
                externas=3, internas=2)
            self.assertEqual(manifesto["tipo_dados"], "synthetic_test")
            self.assertEqual(len(manifesto["divisoes"]), 3)
            self.assertEqual(len(linhas), 9)
            self.assertTrue((destino / "manifesto.json").is_file())
            salvas = [json.loads(l) for l in (destino / "resultados.jsonl").read_text(encoding="utf-8").splitlines()]
            externas = [r for r in salvas if r["etapa"] == "avaliacao_externa"]
            self.assertEqual(len(externas), 3)
            self.assertTrue(all(r["elegivel"] for r in externas))
            self.assertTrue(all(resultados_compativeis(r, r.copy()) for r in externas))
            self.assertFalse(resultados_compativeis(externas[0], externas[1]))
            self.assertEqual(len(externas[0]["previsoes"]), externas[0]["linhas_validacao"])
            self.assertEqual(externas[0]["metricas"],
                             calcular_metricas(dados["class"].to_numpy()[externas[0]["indices_validacao"]],
                                              externas[0]["previsoes"]))

    def test_familias_cpu_compartilham_contrato_de_metricas(self):
        dados = fixture()
        x, y, _, plano = planejar_divisoes(dados, internas=2)
        _, treino, val, _ = plano[0]
        for familia in ("svm_linear", "svm_rbf", "mlp_pequena", "mlp_profunda"):
            parametros = {"epocas": 2} if familia.startswith("mlp_") else {}
            config = Configuracao(familia, familia, parametros,
                                  balanceamento="over_0_5" if familia.startswith("mlp_") else "nenhum")
            with self.subTest(familia=familia):
                resultado = avaliar_divisao(x, y, treino, val, config)
                self.assertEqual(resultado["linhas_validacao"], len(val))
                self.assertEqual(len(resultado["previsoes"]), len(val))
                self.assertEqual(sum(map(sum, resultado["metricas"]["matriz"])), len(val))
                self.assertIn("status_convergencia", resultado)

    def test_selecao_ignora_configuracao_sem_convergencia(self):
        class SVMIncompleta:
            fit_status_ = 1
            def fit(self, x, y):
                return self
            def predict(self, x):
                return np.array(["normal"] * len(x))
        with tempfile.TemporaryDirectory() as pasta:
            _, linhas = executar_avaliacao(
                fixture(), [Configuracao("falha", "svm_rbf")], Path(pasta) / "rodada",
                tipo_dados="synthetic_test", externas=3, internas=2,
                fabrica=lambda _: SVMIncompleta())
            externas = [r for r in linhas if r["etapa"] == "avaliacao_externa"]
            self.assertEqual(len(externas), 3)
            self.assertTrue(all(r["status"] == "sem_configuracao_elegivel" for r in externas))
            self.assertTrue(all(not r["elegivel"] for r in linhas if r["etapa"] == "selecao_interna"))

    def test_tipo_oficial_exige_arquivo_de_treino(self):
        with self.assertRaises(ValueError):
            preparar_entrada(fixture(), "official_training")
        with self.assertRaises(ValueError):
            preparar_entrada("teste_sem_rotulo.csv", "official_training")


if __name__ == "__main__":
    unittest.main()
