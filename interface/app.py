# ============================================================
# interface/app.py
# DataBrief AI — Interface Gradio
# Chat centralizado + CheckboxGroup para classificacao_estrategica
# Gradio 5.x / 6.x compatível
# ============================================================

import gradio as gr
import json
from schemas.models import (
    SessionState, DemandState, ModoExecucao,
    ReadinessStatus, TipoDemanda, ClassificacaoEstrategica
)
from graph.agent import processar_turno, processar_selecao_checkbox

# ────────────────────────────────────────────────────────────
# OPÇÕES DE CLASSIFICAÇÃO ESTRATÉGICA
# ────────────────────────────────────────────────────────────

OPCOES_CLASSIFICACAO = [c.value for c in ClassificacaoEstrategica]

OPCOES_RESULTADO = [
    "Dashboard interativo",
    "Agente automatizado",
    "Pipeline de dados",
    "Tabela Gold",
    "Modelo analítico",
    "Outro",
]

# ────────────────────────────────────────────────────────────
# ESTADO GLOBAL DA SESSÃO
# ────────────────────────────────────────────────────────────

def nova_sessao() -> SessionState:
    sessao = SessionState(modo_execucao=ModoExecucao.GPU_LOCAL)
    sessao.adicionar_demanda(DemandState())
    return sessao


# ────────────────────────────────────────────────────────────
# BARRA DE PROGRESSO
# ────────────────────────────────────────────────────────────

def renderizar_barra_progresso(sessao: SessionState) -> str:
    if not sessao or not sessao.demandas:
        return _barra_html(0, "Aguardando demanda")
    demanda = sessao.demanda_ativa
    if not demanda:
        return _barra_html(0, "Aguardando demanda")
    pct = int(demanda.calcular_completude() * 100)
    label_status = {
        ReadinessStatus.PRONTA:          "Pronta para briefing",
        ReadinessStatus.DISCOVERY:       "Em refinamento",
        ReadinessStatus.ESCLARECIMENTO:  "Aguardando confirmação",
        ReadinessStatus.BLOQUEADA:       "Bloqueada",
    }.get(demanda.readiness, "Em andamento")
    return _barra_html(pct, label_status)


def _barra_html(pct: int, label: str) -> str:
    cor = "#22c55e" if pct == 100 else "#3b82f6"
    return f"""
    <div style="padding: 6px 0 10px 0; font-family: sans-serif;">
        <div style="display: flex; justify-content: space-between;
                    align-items: center; margin-bottom: 5px;">
            <span style="font-size: 12px; color: #6b7280;">{label}</span>
            <span style="font-size: 12px; font-weight: 600; color: {cor};">{pct}%</span>
        </div>
        <div style="background: #e5e7eb; border-radius: 99px; height: 4px;">
            <div style="background: {cor}; width: {pct}%; height: 4px;
                        border-radius: 99px; transition: width 0.4s ease;"></div>
        </div>
    </div>
    """


# ────────────────────────────────────────────────────────────
# BRIEFING FINAL
# ────────────────────────────────────────────────────────────

def renderizar_briefing_final(sessao: SessionState) -> str:
    if not sessao or not sessao.demandas:
        return ""
    demanda = sessao.demanda_ativa
    if not demanda or demanda.readiness != ReadinessStatus.PRONTA:
        return ""
    return _bloco_briefing(demanda)


def _bloco_briefing(demanda) -> str:
    def campo_linha(label, valor, origem=None, turno=None):
        if not valor:
            return ""
        origem_badge = ""
        if origem:
            cor_origem = {
                "TEXT": "#3b82f6", "RULE": "#8b5cf6",
                "AUDIO": "#f59e0b", "RAG": "#22c55e",
                "ATTACHMENT": "#f97316", "MANUAL": "#6b7280",
            }.get(origem, "#9ca3af")
            turno_txt = f" · turno {turno}" if turno is not None else ""
            origem_badge = (
                f'<span style="font-size:10px;background:{cor_origem}18;color:{cor_origem};'
                f'border:1px solid {cor_origem}44;padding:1px 7px;border-radius:99px;'
                f'margin-left:6px;vertical-align:middle;">{origem.lower()}{turno_txt}</span>'
            )
        return f"""
        <div style="display:flex;gap:12px;padding:10px 0;
                    border-bottom:1px solid #f3f4f6;align-items:flex-start;">
            <div style="min-width:150px;font-size:12px;color:#9ca3af;padding-top:2px;">{label}</div>
            <div style="flex:1;font-size:13px;color:#111;font-weight:500;">
                {str(valor)}{origem_badge}
            </div>
        </div>"""

    def fp_linha(label, fp):
        if not fp:
            return ""
        return campo_linha(label, fp.valor,
                           fp.origem.value if hasattr(fp, "origem") else None,
                           fp.turno if hasattr(fp, "turno") else None)

    completude = demanda.calcular_completude()
    pct = int(completude * 100)
    tipo_nome = {
        TipoDemanda.ANALISE: "Análise", TipoDemanda.ESTRUTURANTE: "Estruturante",
        TipoDemanda.PRODUTO_DADOS: "Produto de Dados", TipoDemanda.ALARMASTICA: "Alarmística",
    }.get(demanda.tipo_demanda, "Indefinido")

    pendencias_html = ""
    if demanda.pendencias:
        items = "".join(
            f'<span style="background:#fef3c7;color:#92400e;border:1px solid #fde68a;'
            f'padding:2px 8px;border-radius:99px;font-size:11px;margin:2px;">{p}</span>'
            for p in demanda.pendencias
        )
        pendencias_html = f"""
        <div style="margin-top:12px;">
            <div style="font-size:12px;color:#9ca3af;margin-bottom:6px;">Pendências</div>
            <div style="display:flex;flex-wrap:wrap;gap:4px;">{items}</div>
        </div>"""

    classificacoes = ""
    if demanda.classificacao_estrategica:
        vals = ", ".join(c.value for c in demanda.classificacao_estrategica)
        classificacoes = campo_linha("Classificação estratégica", vals)

    perguntas_html = ""
    if demanda.perguntas_de_negocio:
        perguntas_txt = " | ".join(fp.valor for fp in demanda.perguntas_de_negocio)
        perguntas_html = campo_linha(
            "Perguntas de negócio", perguntas_txt,
            demanda.perguntas_de_negocio[0].origem.value if demanda.perguntas_de_negocio else None,
            demanda.perguntas_de_negocio[0].turno if demanda.perguntas_de_negocio else None,
        )

    return f"""
    <div style="border:2px solid #22c55e;border-radius:12px;padding:24px;
                background:white;font-family:sans-serif;margin-top:8px;">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px;">
            <div>
                <div style="font-size:16px;font-weight:700;color:#111;">Briefing Gerado</div>
                <div style="font-size:12px;color:#6b7280;margin-top:2px;">{tipo_nome} · {pct}% completo</div>
            </div>
            <span style="background:#dcfce7;color:#166534;border:1px solid #bbf7d0;
                         padding:4px 12px;border-radius:99px;font-size:12px;font-weight:600;">
                Pronto para aprovação
            </span>
        </div>
        <div style="border-top:1px solid #f3f4f6;padding-top:4px;">
            {fp_linha("Título", demanda.titulo)}
            {fp_linha("Objetivo", demanda.objetivo)}
            {fp_linha("Resultado esperado", demanda.resultado_esperado)}
            {campo_linha("Valor de negócio", demanda.valor_negocio.value if demanda.valor_negocio else None)}
            {classificacoes}
            {perguntas_html}
            {fp_linha("Bloqueios", demanda.bloqueios)}
        </div>
        {pendencias_html}
    </div>"""


# ────────────────────────────────────────────────────────────
# LÓGICA DO CHAT
# ────────────────────────────────────────────────────────────

def adicionar_mensagem_usuario(mensagem, historico):
    # Passo 1: sobe a mensagem do usuario imediatamente, limpa a caixa
    if not mensagem.strip():
        return historico, mensagem
    historico = historico or []
    historico.append({"role": "user", "content": mensagem})
    return historico, ""


def processar_resposta(historico, sessao_state, modo_str):
    # Passo 2: processa e adiciona a resposta do agente
    _vazio = (historico, sessao_state,
              _barra_html(0, "Aguardando demanda"), "",
              gr.update(visible=False), gr.update(visible=False),
              gr.update(visible=False), gr.update(visible=False),
              gr.update(value=[]), gr.update(visible=False), gr.update(value=None))
    if not historico:
        return _vazio
    mensagem_raw = next(
        (m["content"] for m in reversed(historico) if m["role"] == "user"), ""
    )
    # Gradio 6.x pode retornar content como lista de blocos
    if isinstance(mensagem_raw, list):
        mensagem = " ".join(b.get("text", "") for b in mensagem_raw if isinstance(b, dict))
    else:
        mensagem = str(mensagem_raw)
    if not mensagem.strip():
        return _vazio

    sessao = SessionState.model_validate(sessao_state) if sessao_state else nova_sessao()
    modo_map = {
        "GPU Local (Qwen3-4B)":   ModoExecucao.GPU_LOCAL,
        "CPU Local (Qwen3-1.7B)": ModoExecucao.CPU_LOCAL,
        "OpenAI (gpt-4o-mini)":   ModoExecucao.OPENAI,
    }
    sessao.modo_execucao = modo_map.get(modo_str, ModoExecucao.GPU_LOCAL)

    sessao, resposta, briefing, campo_atual = processar_turno(agente, sessao, mensagem)
    historico.append({"role": "assistant", "content": resposta})

    barra_html    = renderizar_barra_progresso(sessao)
    pronto        = sessao.demanda_ativa and sessao.demanda_ativa.readiness == ReadinessStatus.PRONTA
    briefing_html = renderizar_briefing_final(sessao) if pronto else ""
    pede_classificacao = campo_atual == "classificacao_estrategica"
    pede_resultado     = campo_atual == "resultado_esperado"

    return (
        historico,
        sessao.model_dump(mode="json"),
        barra_html,
        briefing_html,
        gr.update(visible=pronto),
        gr.update(visible=pronto),
        gr.update(visible=False),
        gr.update(visible=pede_classificacao),
        gr.update(value=[]),
        gr.update(visible=pede_resultado),
        gr.update(value=None),
    )


def confirmar_classificacao(selecoes, sessao_state, historico):
    """Chamada quando o usuário confirma a seleção do CheckboxGroup.
    Aplica a seleção e roda o grafo para formular a próxima pergunta.
    """
    if not selecoes or not sessao_state:
        return historico, sessao_state, gr.update(visible=False), gr.update(visible=False)

    sessao = SessionState.model_validate(sessao_state)
    sessao = processar_selecao_checkbox(sessao, selecoes)

    # Registra a seleção no histórico como mensagem do usuário
    texto_confirmado = ", ".join(selecoes)
    historico = historico or []
    historico.append({"role": "user", "content": f"Classificação: {texto_confirmado}"})

    # Roda o grafo com uma mensagem vazia para avançar para a próxima pergunta
    # O grafo vai avaliar o estado atualizado e formular o próximo campo
    sessao, resposta, briefing, campo_atual = processar_turno(agente, sessao, "__checkbox__")

    historico.append({"role": "assistant", "content": resposta})

    pronto = sessao.demanda_ativa and sessao.demanda_ativa.readiness == ReadinessStatus.PRONTA
    briefing_html_val = renderizar_briefing_final(sessao) if pronto else ""
    pede_classificacao = campo_atual == "classificacao_estrategica"

    return (
        historico,
        sessao.model_dump(mode="json"),
        gr.update(visible=False),               # esconde o checkbox
        gr.update(visible=pede_classificacao),  # mostra de novo só se cair no mesmo campo
        briefing_html_val,
        gr.update(visible=pronto),              # secao_briefing
        gr.update(visible=pronto),              # btn_aprovar
        renderizar_barra_progresso(sessao),     # barra atualizada
    )


def confirmar_resultado(selecao, sessao_state, historico):
    """Chamada quando o usuário confirma o Radio de resultado_esperado.
    Aplica o valor direto ao estado e avança o grafo.
    """
    if not selecao or not sessao_state:
        return (historico, sessao_state, gr.update(visible=False), "",
                gr.update(visible=False), gr.update(visible=False),
                _barra_html(0, "Aguardando demanda"))

    from schemas.models import FieldProvenance, OrigemCampo
    sessao = SessionState.model_validate(sessao_state)
    demanda = sessao.demanda_ativa

    demanda.resultado_esperado = FieldProvenance(
        valor=selecao,
        origem=OrigemCampo.MANUAL,
        turno=demanda.turno_atual,
    )
    sessao.demandas[sessao.indice_ativo] = demanda

    historico = historico or []
    historico.append({"role": "user", "content": f"Resultado esperado: {selecao}"})

    sessao, resposta, briefing, campo_atual = processar_turno(agente, sessao, "__radio__")
    historico.append({"role": "assistant", "content": resposta})

    pronto = sessao.demanda_ativa and sessao.demanda_ativa.readiness == ReadinessStatus.PRONTA
    briefing_html_val = renderizar_briefing_final(sessao) if pronto else ""

    return (
        historico,
        sessao.model_dump(mode="json"),
        gr.update(visible=False),
        briefing_html_val,
        gr.update(visible=pronto),
        gr.update(visible=pronto),
        renderizar_barra_progresso(sessao),
    )


def aprovar_briefing(sessao_state):
    if not sessao_state:
        return None, gr.update(visible=False), "Nenhum briefing para aprovar."
    sessao  = SessionState.model_validate(sessao_state)
    demanda = sessao.demanda_ativa
    from schemas.models import BriefingOutput
    briefing = BriefingOutput.from_demand_state(demanda, sessao.modo_execucao)
    dados = briefing.to_sharepoint_dict()
    dados["_metadata"] = {
        "id_demanda":    briefing.id_demanda,
        "sessao_id":     briefing.sessao_id,
        "completude":    briefing.completude,
        "gerado_em":     briefing.gerado_em.isoformat(),
        "modo_execucao": briefing.modo_execucao.value,
        "proveniencia":  {
            k: {"origem": v.origem.value, "turno": v.turno}
            for k, v in briefing.proveniencia.items()
        },
    }
    caminho = f"/tmp/briefing_{briefing.id_demanda}.json"
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(dados, f, ensure_ascii=False, indent=2)
    return caminho, gr.update(visible=True), "Briefing aprovado. Faça o download abaixo."


def reiniciar(historico, sessao_state):
    sessao_limpa = nova_sessao()
    return (
        [], sessao_limpa.model_dump(mode="json"),
        _barra_html(0, "Aguardando demanda"), "",
        gr.update(visible=False), gr.update(visible=False),
        gr.update(visible=False), gr.update(visible=False),
        gr.update(value=[]), "",
    )


# ────────────────────────────────────────────────────────────
# CONSTRUÇÃO DA INTERFACE
# ────────────────────────────────────────────────────────────

def construir_interface(agente_compilado) -> gr.Blocks:
    global agente
    agente = agente_compilado

    css = """
    .gradio-container { max-width: 860px !important; margin: 0 auto !important; }
    .chat-container   { max-width: 780px; margin: 0 auto; width: 100%; }
    """

    with gr.Blocks(title="DataBrief AI") as demo:

        demo.launch.__func__  # apenas referência — css vai no launch()
        sessao_state = gr.State({})

        # ── Cabeçalho ────────────────────────────────────────
        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown("## DataBrief AI\n*Agente para refinamento de demandas de dados*")
            with gr.Column(scale=1, min_width=200):
                modo_selector = gr.Dropdown(
                    choices=["GPU Local (Qwen3-4B)", "CPU Local (Qwen3-1.7B)", "OpenAI (gpt-4o-mini)"],
                    value="GPU Local (Qwen3-4B)",
                    label="Modo",
                )

        gr.HTML("<hr style='margin:4px 0 0 0;border:none;border-top:1px solid #e5e7eb;'>")

        # ── Barra de progresso ────────────────────────────────
        barra_progresso = gr.HTML(value=_barra_html(0, "Aguardando demanda"))

        # ── Chat ─────────────────────────────────────────────
        with gr.Column(elem_classes=["chat-container"]):
            chatbot = gr.Chatbot(
                value=[], height=430, show_label=False, layout="bubble",
                placeholder=(
                    "**Olá! Sou o DataBrief AI.**\n\n"
                    "Descreva sua demanda de dados e vou ajudar a estruturá-la "
                    "em um briefing completo.\n\n"
                    "*Exemplo: \"Preciso de um dashboard de matrículas por polo\"*"
                ),
            )

            # ── CheckboxGroup — classificacao_estrategica ─────
            # Visível apenas quando o agente pede esse campo
            # Fica acima da caixa de texto para aparecer próximo ao chat
            with gr.Column(visible=False) as secao_checkbox:
                gr.Markdown("**Selecione uma ou mais classificações estratégicas:**")
                checkbox_classificacao = gr.CheckboxGroup(
                    choices=OPCOES_CLASSIFICACAO,
                    label="",
                    show_label=False,
                )
                btn_confirmar_classificacao = gr.Button(
                    "Confirmar seleção", variant="primary", size="sm"
                )

            with gr.Column(visible=False) as secao_radio_resultado:
                gr.Markdown("**Como essa demanda será entregue?**")
                radio_resultado = gr.Radio(
                    choices=OPCOES_RESULTADO,
                    label="",
                    show_label=False,
                )
                btn_confirmar_resultado = gr.Button(
                    "Confirmar", variant="primary", size="sm"
                )

            msg_input = gr.Textbox(
                placeholder="Descreva sua demanda...",
                show_label=False, lines=1, max_lines=4, submit_btn=True,
            )

            btn_reiniciar = gr.Button("Nova demanda", size="sm", variant="secondary")

        # ── Briefing final ────────────────────────────────────
        with gr.Column(elem_classes=["chat-container"], visible=False) as secao_briefing:
            briefing_html = gr.HTML(value="")
            btn_aprovar = gr.Button(
                "Aprovar e baixar briefing", variant="primary", size="lg", visible=False
            )
            arquivo_download = gr.File(
                label="Download do briefing (JSON)", visible=False, file_types=[".json"]
            )
            status_aprovacao = gr.Markdown("")

        # ── Eventos ──────────────────────────────────────────
        msg_input.submit(
            fn=adicionar_mensagem_usuario,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, msg_input],
        ).then(
            fn=processar_resposta,
            inputs=[chatbot, sessao_state, modo_selector],
            outputs=[
                chatbot, sessao_state, barra_progresso, briefing_html,
                secao_briefing, btn_aprovar, arquivo_download,
                secao_checkbox, checkbox_classificacao,
                secao_radio_resultado, radio_resultado,
            ],
        )

        btn_confirmar_classificacao.click(
            fn=confirmar_classificacao,
            inputs=[checkbox_classificacao, sessao_state, chatbot],
            outputs=[
                chatbot, sessao_state,
                secao_checkbox, secao_checkbox,   # esconde e controla reexibição
                briefing_html,
                secao_briefing, btn_aprovar,
                barra_progresso,
            ],
        )

        btn_confirmar_resultado.click(
            fn=confirmar_resultado,
            inputs=[radio_resultado, sessao_state, chatbot],
            outputs=[
                chatbot, sessao_state,
                secao_radio_resultado,
                briefing_html,
                secao_briefing, btn_aprovar,
                barra_progresso,
            ],
        )

        btn_aprovar.click(
            fn=aprovar_briefing,
            inputs=[sessao_state],
            outputs=[arquivo_download, arquivo_download, status_aprovacao],
        )

        btn_reiniciar.click(
            fn=reiniciar,
            inputs=[chatbot, sessao_state],
            outputs=[
                chatbot, sessao_state, barra_progresso, briefing_html,
                secao_briefing, btn_aprovar, arquivo_download,
                secao_checkbox, checkbox_classificacao, status_aprovacao,
            ],
        )

    return demo
