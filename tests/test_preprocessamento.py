import unittest
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from preprocessamento import (
    analisar_duplicatas, analisar_violacoes_dominio, encontrar_inconsistencias_caixa,
    investigar_su_attempted, resumir_taxas_invalidas, verificar_redundancias,
    analisar_faixas_referencia, impacto_dos_conflitos,
    analisar_associacoes_com_classe, resumir_ausencias_por_classe, carregar_treino,
)


class DiagnosticoTest(unittest.TestCase):
    def setUp(self):
        self.dados = pd.DataFrame({"duration":[1,-1,1],"src_bytes":[10,-2,10],"dst_bytes":[0,0,0],"rerror_rate":[.5,2,.5],"count":[1,512,1],"srv_count":[1,1,1],"dst_host_count":[1,1,1],"su_attempted":[0,2,0],"protocol_type":["tcp","TCP","tcp"],"service":["http","telnet","http"],"root_shell":[0,1,0],"class":["normal","anomaly","normal"]})

    def test_violacoes_e_su_attempted(self):
        tabela = analisar_violacoes_dominio(self.dados).set_index("atributo")
        self.assertEqual(tabela.loc["duration", "violacoes"], 1)
        self.assertEqual(tabela.loc["rerror_rate", "violacoes"], 1)
        self.assertEqual(investigar_su_attempted(self.dados)["registros_com_valor_2"].shape[0], 1)

    def test_caixa_e_duplicatas(self):
        self.assertEqual(encontrar_inconsistencias_caixa(self.dados,["protocol_type"]).iloc[0]["atributo"],"protocol_type")
        resumo, conflitos = analisar_duplicatas(self.dados)
        self.assertEqual(resumo.iloc[0]["quantidade"], 1)
        self.assertTrue(conflitos.empty)

    def test_preserva_espacos_e_agrupa_variantes_normalizadas(self):
        dados = pd.DataFrame({"protocol_type":["tcp","TCP"," tcp ","udp"], "class":["normal"] * 4})
        tabela = encontrar_inconsistencias_caixa(dados, ["protocol_type"])
        self.assertEqual(tabela.iloc[0]["quantidade_variantes"], 3)
        self.assertIn(" tcp ", tabela.iloc[0]["variantes_encontradas"])
        self.assertEqual(resumir_taxas_invalidas(self.dados)["registros_com_taxa_invalida"], 1)

    def test_redundancias(self):
        d = pd.DataFrame({"src_bytes":[1024.],"dst_bytes":[2048.],"same_srv_rate":[.5],"src_bytes_kb":[1.],"dst_bytes_kb":[2.],"total_bytes":[3072.],"same_srv_pct":[50.],"protocol_type":["tcp"],"protocolo_cod":[0],"class":["normal"]})
        self.assertTrue(verificar_redundancias(d)["redundancia_comprovada"].all())

    def test_faixa_indicativa_nao_e_apresentada_como_violacao_de_dominio(self):
        dominios = analisar_violacoes_dominio(self.dados)
        referencias = analisar_faixas_referencia(self.dados).set_index("atributo")
        self.assertNotIn("count", dominios["atributo"].tolist())
        self.assertEqual(referencias.loc["count", "ocorrencias_acima"], 1)

    def test_diferencia_repeticoes_de_conflitos_e_preserva_indices(self):
        dados = pd.DataFrame({"valor": [10, 10, 20, 20, 30],
                              "class": ["normal", "normal", "normal", "anomaly", "anomaly"]},
                             index=[8, 2, 7, 4, 9])
        original = dados.copy(deep=True)
        resumo, conflitos = analisar_duplicatas(dados)
        self.assertEqual(resumo["quantidade"].tolist(), [1, 2, 2])
        self.assertEqual(conflitos.index.tolist(), [7, 4])
        impacto = impacto_dos_conflitos(dados, conflitos).set_index("classe")
        self.assertEqual(impacto.loc["anomaly", "percentual_da_classe_envolvido"], 50)
        self.assertEqual(impacto.loc["normal", "registros_envolvidos"], 1)
        assert_frame_equal(dados, original)

    def test_medidas_de_associacao_e_constante(self):
        dados = pd.DataFrame({"numero": [0, 0, 1, 1], "categoria": ["a", "a", "b", "b"],
                              "protocolo_cod": [2, 1, 2, 1], "constante": [0] * 4,
                              "class": ["normal", "normal", "anomaly", "anomaly"]})
        original = dados.copy(deep=True)
        tabela = analisar_associacoes_com_classe(dados).set_index("atributo")
        self.assertAlmostEqual(tabela.loc["numero", "valor"], 1)
        self.assertAlmostEqual(tabela.loc["categoria", "valor"], 1)
        self.assertEqual(tabela.loc["protocolo_cod", "medida"], "Cramer V")
        self.assertAlmostEqual(tabela.loc["protocolo_cod", "valor"], 0)
        self.assertTrue(pd.isna(tabela.loc["constante", "valor"]))
        assert_frame_equal(dados, original)

    def test_cramer_v_reproduz_exemplo_publicado_pelo_scipy(self):
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.contingency.association.html
        observados = [[100, 150], [203, 322], [420, 700], [320, 210]]
        categorias, classes = [], []
        for grupo, contagens in enumerate(observados):
            for classe, quantidade in zip(["normal", "anomaly"], contagens):
                categorias.extend([str(grupo)] * quantidade)
                classes.extend([classe] * quantidade)
        dados = pd.DataFrame({"categoria": categorias, "class": classes})
        valor = analisar_associacoes_com_classe(dados).iloc[0]["valor"]
        self.assertAlmostEqual(valor, 0.18617813077483678)

    def test_ausencias_por_classe_nao_imputam_dados(self):
        dados = pd.DataFrame({"tempo": [np.nan, np.nan, 5, np.nan],
                              "class": ["normal", "normal", "anomaly", "anomaly"]})
        original = dados.copy(deep=True)
        resumo = resumir_ausencias_por_classe(dados, ["tempo"]).set_index("classe")
        self.assertEqual(resumo.loc["normal", "percentual_ausente"], 100)
        self.assertEqual(resumo.loc["anomaly", "percentual_ausente"], 50)
        assert_frame_equal(dados, original)

    def test_leitura_rejeita_classe_fora_do_formato_oficial(self):
        with tempfile.TemporaryDirectory() as diretorio:
            caminho = Path(diretorio) / "treino.csv"
            pd.DataFrame({"valor": [1, 2], "class": ["normal", "anomaly"]}).to_csv(caminho, index=False)
            self.assertEqual(len(carregar_treino(caminho)), 2)
            pd.DataFrame({"valor": [1], "class": ["ataque"]}).to_csv(caminho, index=False)
            with self.assertRaisesRegex(ValueError, "normal ou anomaly"):
                carregar_treino(caminho)
