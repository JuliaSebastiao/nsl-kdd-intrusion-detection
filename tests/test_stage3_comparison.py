"""Invariantes da comparacao e retomada, exclusivamente com fixtures sinteticas."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from test_stage3_evaluation import fixture
from stage3_evaluation import Configuracao, avaliar_divisao, executar_avaliacao, planejar_divisoes
from stage3_comparison import consolidar, gerar_relatorio, plano_experimentos, verificar_base


class ComparacaoTest(unittest.TestCase):
    def executar(self, destino, **kwargs):
        config = Configuracao("rf", "random_forest", {"n_estimators": 3})
        return executar_avaliacao(fixture(), [config], destino, tipo_dados="synthetic_test",
                                  externas=3, internas=2, **kwargs)

    def test_retomada_nao_repete_ajustes_e_recupera_fragmento(self):
        with tempfile.TemporaryDirectory() as p:
            destino=Path(p)/"run"
            _, original=self.executar(destino)
            arquivo=destino/"resultados.jsonl"
            linhas=arquivo.read_text(encoding="utf-8").splitlines()
            arquivo.write_text("\n".join(linhas[:-1])+"\n{\"parcial\":",encoding="utf-8")
            _, retomadas=self.executar(destino,retomar=True)
            self.assertEqual(len(original),len(retomadas))
            self.assertEqual(original[-1]["previsoes"],retomadas[-1]["previsoes"])
            self.assertEqual(original[:-1],retomadas[:-1])
            self.assertTrue(arquivo.with_suffix(".partial").exists())
            def proibido(_):
                raise AssertionError("Treinamento repetido")
            self.executar(destino,retomar=True,fabrica=proibido)

    def test_retomada_rejeita_configuracao_alterada(self):
        with tempfile.TemporaryDirectory() as p:
            destino=Path(p)/"run"
            self.executar(destino)
            manifesto=destino/"manifesto.json"
            d=json.loads(manifesto.read_text(encoding="utf-8"))
            d["fontes_sha256"]["stage3_evaluation.py"]="alterado"
            manifesto.write_text(json.dumps(d),encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"Retomada incompativel"):
                self.executar(destino,retomar=True)

    def test_consolidacao_verifica_metricas_e_cobertura(self):
        with tempfile.TemporaryDirectory() as p:
            destino=Path(p)/"run"
            self.executar(destino)
            _,folds,resumo,configs,selecao=consolidar(destino)
            self.assertEqual(len(folds),3)
            self.assertTrue(resumo.iloc[0].elegivel)
            self.assertEqual(np.asarray(resumo.iloc[0].matriz_oof).sum(),len(fixture()))
            self.assertAlmostEqual(resumo.iloc[0].f1_macro_desvio,folds.f1_macro.std(ddof=1))
            self.assertEqual(selecao["tipo_dados"],"synthetic_test")
            self.assertEqual(configs.iloc[0].ajustes,6)
            arquivo=destino/"resultados.jsonl"
            registros=[json.loads(l) for l in arquivo.read_text(encoding="utf-8").splitlines()]
            registros[0]["metricas"]["matriz"][0][0]+=1
            arquivo.write_text("".join(json.dumps(r)+"\n" for r in registros),encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"Metricas"):
                consolidar(destino)

    def test_consolidacao_incompleta_nao_seleciona_vencedor(self):
        with tempfile.TemporaryDirectory() as p:
            destino=Path(p)/"run"
            self.executar(destino)
            arquivo=destino/"resultados.jsonl"
            linhas=arquivo.read_text(encoding="utf-8").splitlines()
            arquivo.write_text("\n".join(linhas[:-1])+"\n",encoding="utf-8")
            _,_,resumo,_,selecao=consolidar(destino)
            self.assertFalse(resumo.iloc[0].elegivel)
            self.assertIsNone(selecao["comparacao_selecionada"])

    def test_comparacoes_reutilizam_ajuste_externo_identico(self):
        with tempfile.TemporaryDirectory() as p:
            _,linhas=self.executar(Path(p)/"run",comparacoes={"A":["rf"],"B":["rf"]})
            externas=[r for r in linhas if r["etapa"]=="avaliacao_externa"]
            self.assertEqual(len(externas),3)
            self.assertTrue(all(r["comparacoes"]==["A","B"] for r in externas))
            _,folds,resumo,_,_=consolidar(Path(p)/"run")
            self.assertEqual(len(folds),6)
            self.assertTrue(resumo.elegivel.all())

    def test_under_e_pca_preservam_validacao(self):
        x,y,_,plano=planejar_divisoes(fixture(),externas=3,internas=2)
        _,a,v,_=plano[0]
        base=Configuracao("rf","random_forest",{"n_estimators":3})
        antes=avaliar_divisao(x,y,a,v,base)
        depois=avaliar_divisao(x,y,a,v,replace(base,balanceamento="under_0_5",reducao="pca60"))
        self.assertEqual(antes["indices_validacao"],depois["indices_validacao"])
        self.assertEqual(antes["rotulos_reais"],depois["rotulos_reais"])
        self.assertLess(depois["linhas_ajuste"],antes["linhas_ajuste"])
        self.assertLessEqual(depois["colunas"],60)

    def test_plano_cobre_tecnicas_e_preserva_epocas(self):
        configs,comparacoes=plano_experimentos()
        self.assertEqual(len(configs),15)
        self.assertEqual({c.balanceamento for c in configs},{"nenhum","over_0_5","under_0_5"})
        self.assertEqual({c.parametros["epocas"] for c in configs if c.modelo.startswith("mlp")},{120})
        self.assertEqual(len(comparacoes["Random Forest"]),4)

    def test_esquema_oficial_nao_aceita_fixture_renomeada(self):
        with tempfile.TemporaryDirectory() as p:
            arquivo=Path(p)/"treino.csv"
            fixture().to_csv(arquivo,index=False)
            with self.assertRaisesRegex(ValueError,"Esquema diferente"):
                verificar_base(arquivo)

    def test_relatorio_separa_agregados_de_registros_privados(self):
        with tempfile.TemporaryDirectory() as p:
            destino=Path(p)/"run"
            self.executar(destino,comparacoes={"Random Forest":["rf"],"preparo:referencia":["rf"]})
            saida=destino/"resumo"
            gerar_relatorio(destino,saida)
            publico=json.loads((saida/"resultados.json").read_text(encoding="utf-8"))
            self.assertEqual(publico["tipo_dados"],"synthetic_test")
            self.assertNotIn("divisoes",publico)
            self.assertNotIn("origem",publico)
            self.assertEqual(len(list(saida.glob("*.png"))),5)
            self.assertEqual(len(list(saida.glob("*.csv"))),3)
            texto=json.dumps(publico)
            for chave in ("indices_validacao", "rotulos_reais", "previsoes"):
                self.assertNotIn(chave,texto)


if __name__=="__main__":
    unittest.main()
