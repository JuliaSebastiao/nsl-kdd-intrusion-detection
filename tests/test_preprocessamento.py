"""Verificações de alinhamento, preservação e isolamento do treino."""

import unittest

import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal

from preprocessamento import aplicar_estrategia, construir_alvo, identificar_conflitos


class LimpezaDeConflitosTest(unittest.TestCase):
    def test_remove_todo_grupo_conflitante_e_preserva_repeticoes_coerentes(self):
        X = pd.DataFrame({"atributo": [10, 10, 20, 20, 30]}, index=[8, 2, 7, 4, 9])
        y = pd.Series(["A", "B", "C", "C", "D"], index=X.index)
        X_antes, y_antes = X.copy(), y.copy()
        limpo, alvo, removidos = aplicar_estrategia(X, y, "remover_conflitos")
        self.assertEqual(list(removidos), [8, 2])
        self.assertEqual(list(limpo.index), [7, 4, 9])
        assert_frame_equal(limpo, X.loc[[7, 4, 9]])
        assert_series_equal(alvo, y.loc[[7, 4, 9]])
        assert_frame_equal(X, X_antes)
        assert_series_equal(y, y_antes)

    def test_rotulo_na_validacao_nao_participa_da_decisao_de_limpeza(self):
        X_treino = pd.DataFrame({"atributo": [10, 20]}, index=[1, 2])
        y_treino = pd.Series(["A", "B"], index=X_treino.index)
        X_validacao = pd.DataFrame({"atributo": [10]}, index=[3])
        y_validacao = pd.Series(["B"], index=X_validacao.index)
        X_validacao_antes, y_validacao_antes = X_validacao.copy(), y_validacao.copy()
        limpo, alvo, removidos = aplicar_estrategia(X_treino, y_treino, "remover_conflitos")
        self.assertEqual(len(removidos), 0)
        assert_frame_equal(limpo, X_treino)
        assert_series_equal(alvo, y_treino)
        assert_frame_equal(X_validacao, X_validacao_antes)
        assert_series_equal(y_validacao, y_validacao_antes)

    def test_mudanca_do_alvo_pode_resolver_conflito(self):
        dados = pd.DataFrame({"atributo": [10, 10], "attack_type": ["nmap", "ipsweep"]})
        X = dados[["atributo"]]
        self.assertTrue(identificar_conflitos(X, construir_alvo(dados, "original")).all())
        self.assertFalse(identificar_conflitos(X, construir_alvo(dados, "familias")).any())
        self.assertEqual(list(dados["attack_type"]), ["nmap", "ipsweep"])

    def test_rejeita_alvo_com_indices_fora_de_ordem(self):
        X = pd.DataFrame({"atributo": [10, 20]}, index=[1, 2])
        y = pd.Series(["B", "A"], index=[2, 1])
        with self.assertRaisesRegex(ValueError, "mesma ordem"):
            aplicar_estrategia(X, y, "remover_conflitos")

    def test_rejeita_rotulo_ou_dificuldade_na_chave(self):
        for coluna in ["attack_type", "difficulty"]:
            with self.subTest(coluna=coluna):
                X = pd.DataFrame({"atributo": [10], coluna: [1]})
                y = pd.Series(["A"])
                with self.assertRaisesRegex(ValueError, "metadado"):
                    identificar_conflitos(X, y)

    def test_manter_entrega_copias_independentes(self):
        X = pd.DataFrame({"atributo": [10, 10]}, index=[1, 2])
        y = pd.Series(["A", "B"], index=X.index)
        copia, alvo, removidos = aplicar_estrategia(X, y, "manter")
        self.assertEqual(len(removidos), 0)
        copia.loc[1, "atributo"] = 99
        alvo.loc[1] = "C"
        self.assertEqual(X.loc[1, "atributo"], 10)
        self.assertEqual(y.loc[1], "A")


if __name__ == "__main__":
    unittest.main()
