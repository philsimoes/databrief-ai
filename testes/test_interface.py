# ============================================================
# testes/test_interface.py
# DataBrief AI — Bloco 32: testes de regressão da interface (Bloco 31)
# Confere que cada função devolve o mesmo número de valores que a lista
# outputs= do Gradio espera — o tipo de bug que já apareceu no fechamento
# do Ato 2 (tupla com 13 valores contra 15 esperados).
# ============================================================

import json

from schemas.models import (
    DemandState, FieldProvenance, OrigemCampo, ReadinessStatus, SessionState,
)
import graph.agent as ag


def _sessao_pronta():
    s = SessionState()
    s.adicionar_demanda(DemandState())
    s = ag.processar_confirmacao_tipo_demanda(s, "Análise")
    s = ag.processar_confirmacao_valor_negocio(s, "Tático")
    s = ag.processar_selecao_checkbox(s, ["Insight para Decisão"])
    d = s.demanda_ativa
    d.titulo = FieldProvenance(valor="Evasão", origem=OrigemCampo.RULE, turno=4)
    d.objetivo = FieldProvenance(valor="Entender evasão", origem=OrigemCampo.TEXT, turno=1)
    d.resultado_esperado = FieldProvenance(
        valor="Análise: Entender evasão", origem=OrigemCampo.RULE, turno=2)
    d.perguntas_de_negocio = [FieldProvenance(valor="Por quê?", origem=OrigemCampo.MANUAL, turno=4)]
    d.readiness = ReadinessStatus.PRONTA
    return s.model_dump(mode="json")


def _contagem_outputs(demo, nome_funcao):
    fns = demo.fns.values() if isinstance(demo.fns, dict) else demo.fns
    return [len(f.outputs) for f in fns if getattr(f.fn, "__name__", "") == nome_funcao]


def test_aridade_dos_eventos_bate(app):
    demo = app.construir_interface(None)
    assert _contagem_outputs(demo, "reiniciar") == [len(app.reiniciar([], {}))]
    assert _contagem_outputs(demo, "aprovar_briefing") == [len(app.aprovar_briefing(_sessao_pronta()))]
    assert _contagem_outputs(demo, "abrir_edicao") == [len(app.abrir_edicao(_sessao_pronta()))]


def test_painel_mostra_origem_e_rotulo_ia(app):
    html = app.renderizar_briefing_final(SessionState.model_validate(_sessao_pronta()))
    assert "Tipo de demanda" in html
    assert "selecionado" in html          # badge MANUAL dos campos de Radio
    assert "inferido pelo sistema" in html
    assert "apoio de IA" in html


def test_aprovar_com_edicao_vira_manual(app):
    saida = app.aprovar_briefing(_sessao_pronta(), "Evasão no Digital — Estácio", "Entender evasão")
    dados = json.load(open(saida[0], encoding="utf-8"))
    prov = dados["_metadata"]["proveniencia"]
    assert dados["Título"] == "Evasão no Digital — Estácio"
    assert prov["titulo"]["origem"] == "MANUAL"
    assert prov["objetivo"]["origem"] == "TEXT"   # não mudou → origem original
    assert all(v["confirmado"] for v in prov.values())
    assert "aviso_ia" in dados["_metadata"]


def test_aprovar_sem_edicao_mantem_valores(app):
    saida = app.aprovar_briefing(_sessao_pronta(), "", "")
    assert json.load(open(saida[0], encoding="utf-8"))["Título"] == "Evasão"


def test_editar_objetivo_de_analise_reaplica_regra(app):
    saida = app.aprovar_briefing(_sessao_pronta(), "", "Entender a evasão do Digital")
    assert json.load(open(saida[0], encoding="utf-8"))["Resultado"] == "Análise: Entender a evasão do Digital"
