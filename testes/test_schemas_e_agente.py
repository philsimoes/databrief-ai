# ============================================================
# testes/test_schemas_e_agente.py
# DataBrief AI — Bloco 32: testes de regressão (schemas + regras do agente)
# Nenhum teste chama o LLM — só lógica determinística.
# ============================================================

import schemas.models as m
from schemas.models import (
    BriefingOutput, DemandState, FieldProvenance, ModoExecucao, OrigemCampo,
    SessionState, TipoDemanda,
)
import graph.agent as ag


def _sessao():
    s = SessionState()
    s.adicionar_demanda(DemandState())
    return s


# ── Bloco 30: proveniência dos 3 campos de opção fixa ────────

def test_extracao_registra_origem_do_turno():
    d = DemandState()
    d = ag.aplicar_extracao(
        d, {"tipo_demanda": "Produto de Dados", "classificacao_estrategica": ["Monitoramento"]},
        1, "", "quero um dashboard de matrículas por polo", OrigemCampo.AUDIO,
    )
    assert d.tipo_demanda == TipoDemanda.PRODUTO_DADOS
    assert d.proveniencia_campos_fixos["tipo_demanda"].origem == OrigemCampo.AUDIO
    assert d.proveniencia_campos_fixos["classificacao_estrategica"].valor == ["Monitoramento"]


def test_radio_e_checkbox_registram_manual():
    s = _sessao()
    s.demanda_ativa.turno_atual = 3
    s = ag.processar_confirmacao_tipo_demanda(s, "Alarmística")
    s = ag.processar_confirmacao_valor_negocio(s, "Tático")
    s = ag.processar_selecao_checkbox(s, ["Priorização", "Monitoramento"])
    prov = s.demanda_ativa.proveniencia_campos_fixos
    for campo in ["tipo_demanda", "valor_negocio", "classificacao_estrategica"]:
        assert prov[campo].origem == OrigemCampo.MANUAL
    assert prov["tipo_demanda"].turno == 3


def test_valor_invalido_nao_grava_proveniencia():
    s = ag.processar_confirmacao_valor_negocio(_sessao(), "valor-que-nao-existe")
    assert "valor_negocio" not in s.demanda_ativa.proveniencia_campos_fixos


def test_proveniencia_sobrevive_ida_e_volta_json():
    """O Gradio guarda a sessão como JSON entre cliques — não pode perder nada."""
    s = ag.processar_confirmacao_valor_negocio(_sessao(), "Tático")
    s2 = SessionState.model_validate(s.model_dump(mode="json"))
    assert s2.demanda_ativa.proveniencia_campos_fixos["valor_negocio"].valor == "Tático"


def test_sessao_antiga_sem_campo_novo_ainda_carrega():
    dados = _sessao().model_dump(mode="json")
    del dados["demandas"][0]["proveniencia_campos_fixos"]
    SessionState.model_validate(dados)


def test_briefing_exporta_proveniencia_completa():
    s = _sessao()
    s = ag.processar_confirmacao_tipo_demanda(s, "Análise")
    s = ag.processar_confirmacao_valor_negocio(s, "Tático")
    s = ag.processar_selecao_checkbox(s, ["Insight para Decisão"])
    d = s.demanda_ativa
    d.titulo = FieldProvenance(valor="T", origem=OrigemCampo.RULE, turno=4)
    d.objetivo = FieldProvenance(valor="O", origem=OrigemCampo.TEXT, turno=1)
    d.resultado_esperado = FieldProvenance(valor="Análise: O", origem=OrigemCampo.RULE, turno=2)
    d.perguntas_de_negocio = [FieldProvenance(valor="P?", origem=OrigemCampo.MANUAL, turno=4)]
    b = BriefingOutput.from_demand_state(d, ModoExecucao.GPU_LOCAL)
    esperados = {"titulo", "objetivo", "resultado_esperado", "perguntas_de_negocio",
                 "tipo_demanda", "valor_negocio", "classificacao_estrategica"}
    assert esperados <= set(b.proveniencia)


# ── Bloco 30: schema morto removido ──────────────────────────

def test_schema_morto_removido():
    for nome in ["CamposAnalise", "CamposEstruturante", "CamposProdutoDados",
                 "CamposAlarmastica", "analise_depende_de_gold",
                 "DEPENDE_SEMPRE_DE_ESTRUTURANTE"]:
        assert not hasattr(m, nome), nome


# ── Guardas anti-alucinação (Blocos 18 e 21) ─────────────────

def test_mensagem_de_desistencia_nao_reabre_guarda_do_titulo():
    d = ag.aplicar_extracao(
        DemandState(), {"titulo": "blz"}, 2,
        "Não consegui fechar sozinho: titulo. Pode me mandar quando tiver.", "blz",
    )
    assert d.titulo is None


def test_pergunta_real_de_titulo_aceita():
    d = ag.aplicar_extracao(
        DemandState(), {"titulo": "Evasão Digital"}, 2,
        "Como você chamaria essa demanda?", "Evasão Digital",
    )
    assert d.titulo.valor == "Evasão Digital"


def test_frase_de_transmissao_nao_grava_tipo():
    d = ag.aplicar_extracao(
        DemandState(), {"tipo_demanda": "Alarmística"}, 1, "",
        "Segue o briefing que recebi do time comercial do Ibmec, anexei o documento.",
    )
    assert d.tipo_demanda is None
    assert not d.proveniencia_campos_fixos


def test_alerta_mapeia_para_alerta_automatico():
    d = ag.aplicar_extracao(
        DemandState(), {"resultado_esperado": "x"}, 1, "",
        "quero um alerta automático quando a evasão subir",
    )
    assert d.resultado_esperado.valor == "Alerta automático"


# ── Bloco 15: nunca fallback silencioso de modo ──────────────

def test_inicializar_modelo_rejeita_modo_em_texto():
    import pytest
    with pytest.raises(TypeError):
        ag.inicializar_modelo(object(), object(), "CPU_LOCAL")
    ag.inicializar_modelo(object(), object(), ModoExecucao.GPU_LOCAL)
