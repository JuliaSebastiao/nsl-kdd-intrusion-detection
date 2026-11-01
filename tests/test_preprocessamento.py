import unittest
import pandas as pd
from preprocessamento import analisar_duplicatas, analisar_violacoes_dominio, encontrar_inconsistencias_caixa, investigar_su_attempted, resumir_taxas_invalidas, verificar_redundancias


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
