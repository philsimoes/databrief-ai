# ============================================================
# interface/app.py
# DataBrief AI — Bloco 03: Interface Gradio
# Split screen: chat (esq) + briefing ao vivo (dir)
# Gradio 5.x compatível
# ============================================================

import gradio as gr
import json
from datetime import datetime
from schemas.models import (
    SessionState, DemandState, ModoExecucao,
    ReadinessStatus, TipoDemanda
)
from graph.agent import processar_turno

# ────────────────────────────────────────────────────────────
# ESTADO GLOBAL DA SESSÃO
# ────────────────────────────────────────────────────────────

def nova_sessao() -> SessionState:
    sessao = SessionState(modo_execucao=ModoExecucao.GPU_LOCAL)
    sessao.adicionar_demanda(DemandState())
    return sessao


# ────────────────────────────────────────────────────────────
# RENDERIZAÇÃO DO PAINEL DIREITO
# ────────────────────────────────────────────────────────────

ICONE_ORIGEM = {
    "TEXT":       "✏️",
    "AUDIO":      "🎙️",
    "ATTACHMENT": "📎",
    "RAG":        "📚",
    "RULE":       "⚙️",
    "MANUAL":     "👤",
}

def renderizar_painel(sessao: SessionState) -> str:
    if not sessao or not sessao.demandas:
        return _painel_vazio()
    blocos = []
    for i, demanda in enumerate(sessao.demandas):
        ativa = i == sessao.indice_ativo
        blocos.append(_bloco_demanda(demanda, i + 1, ativa))
    return "\n".join(blocos)


def _painel_vazio() -> str:
    return """
    <div style="
        display: flex; flex-direction: column; align-items: center;
        justify-content: center; height: 100%; min-height: 400px;
        color: #9ca3af; font-family: sans-serif;
    ">
        <div style="font-size: 48px; margin-bottom: 16px;">📋</div>
        <div style="font-size: 15px;">
            O briefing aparece aqui conforme a conversa avança
        </div>
    </div>
    """


def _bloco_demanda(demanda, numero: int, ativa: bool) -> str:
    completude = demanda.calcular_completude()
    barra_cor = "#22c55e" if completude == 1.0 else "#3b82f6"
    barra_pct = int(completude * 100)

    status_label, status_cor = {
        ReadinessStatus.PRONTA:          ("✅ Pronta",                "#22c55e"),
        ReadinessStatus.DISCOVERY:       ("🔍 Em refinamento",        "#3b82f6"),
        ReadinessStatus.ESCLARECIMENTO:  ("💬 Aguardando confirmação","#f59e0b"),
        ReadinessStatus.BLOQUEADA:       ("🚫 Bloqueada",             "#ef4444"),
    }.get(demanda.readiness, ("❓ Indefinido", "#6b7280"))

    tipo_cor, tipo_nome = {
        TipoDemanda.ANALISE:       ("#8b5cf6", "Análise"),
        TipoDemanda.ESTRUTURANTE:  ("#3b82f6", "Estruturante"),
        TipoDemanda.PRODUTO_DADOS: ("#22c55e", "Produto de Dados"),
        TipoDemanda.ALARMASTICA:   ("#ef4444", "Alarmística"),
    }.get(demanda.tipo_demanda, ("#6b7280", "Indefinido"))

    borda = "2px solid #3b82f6" if ativa else "1px solid #e5e7eb"

    return f"""
    <div style="
        border: {borda}; border-radius: 12px; padding: 20px;
        margin-bottom: 16px; background: white; font-family: sans-serif;
    ">
        <div style="display:flex; justify-content:space-between;
                    align-items:center; margin-bottom:12px;">
            <div style="display:flex; align-items:center; gap:8px;">
                <span style="font-weight:700; font-size:15px;">Demanda {numero}</span>
                <span style="
                    background:{tipo_cor}22; color:{tipo_cor};
                    border:1px solid {tipo_cor}44;
                    padding:2px 10px; border-radius:99px;
                    font-size:12px; font-weight:600;
                ">{tipo_nome}</span>
            </div>
            <span style="color:{status_cor}; font-size:13px; font-weight:500;">
                {status_label}
            </span>
        </div>

        <div style="margin-bottom:16px;">
            <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                <span style="font-size:12px; color:#6b7280;">Completude</span>
                <span style="font-size:12px; font-weight:600; color:{barra_cor};">
                    {barra_pct}%
                </span>
            </div>
            <div style="background:#e5e7eb; border-radius:99px; height:6px;">
                <div style="
                    background:{barra_cor}; width:{barra_pct}%;
                    height:6px; border-radius:99px;
                "></div>
            </div>
        </div>

        {_renderizar_campos(demanda)}
        {_renderizar_pendencias(demanda)}
    </div>
    """


def _renderizar_campos(demanda) -> str:
    linhas = []

    def linha(label, fp, valor_override=None):
        valor = valor_override or (fp.valor if fp else None)
        if not valor:
            return
        origem = fp.origem.value if fp and hasattr(fp, 'origem') else "RULE"
        icone = ICONE_ORIGEM.get(origem, "•")
        turno = f"turno {fp.turno}" if fp and hasattr(fp, 'turno') else ""
        valor_str = str(valor)
        valor_exibido = valor_str[:120] + "..." if len(valor_str) > 120 else valor_str
        linhas.append(f"""
        <div style="display:flex; gap:10px; padding:8px 0;
                    border-bottom:1px solid #f3f4f6;">
            <div style="min-width:130px; font-size:12px; color:#9ca3af;">
                {label}
            </div>
            <div style="flex:1;">
                <div style="font-size:13px; font-weight:500; color:#111;">
                    {valor_exibido}
                </div>
                <div style="font-size:11px; color:#9ca3af; margin-top:2px;">
                    {icone} {origem.lower()} · {turno}
                </div>
            </div>
        </div>
        """)

    linha("Título",            demanda.titulo)
    linha("Objetivo",          demanda.objetivo)
    linha("Resultado esperado",demanda.resultado_esperado)
    linha("Valor de negócio",  None,
          demanda.valor_negocio.value if demanda.valor_negocio else None)

    if demanda.classificacao_estrategica:
        vals = ", ".join(c.value for c in demanda.classificacao_estrategica)
        linhas.append(f"""
        <div style="display:flex; gap:10px; padding:8px 0;
                    border-bottom:1px solid #f3f4f6;">
            <div style="min-width:130px; font-size:12px; color:#9ca3af;">
                Classificação
            </div>
            <div style="font-size:13px; font-weight:500;">{vals}</div>
        </div>
        """)

    if demanda.perguntas_de_negocio:
        perguntas = "<br>".join(
            f"• {fp.valor}" for fp in demanda.perguntas_de_negocio
        )
        linhas.append(f"""
        <div style="display:flex; gap:10px; padding:8px 0;
                    border-bottom:1px solid #f3f4f6;">
            <div style="min-width:130px; font-size:12px; color:#9ca3af;">
                Perguntas
            </div>
            <div style="font-size:13px;">{perguntas}</div>
        </div>
        """)

    if demanda.bloqueios:
        linha("Bloqueios", demanda.bloqueios)

    if not linhas:
        return '<div style="font-size:13px; color:#9ca3af; padding:8px 0;">Nenhum campo preenchido ainda.</div>'

    return f'<div style="margin-bottom:12px;">{"".join(linhas)}</div>'


def _renderizar_pendencias(demanda) -> str:
    if not demanda.pendencias:
        return ""
    items = "".join(
        f'<span style="background:#fef3c7; color:#92400e; '
        f'border:1px solid #fde68a; padding:2px 8px; '
        f'border-radius:99px; font-size:11px; margin:2px;">'
        f'{p}</span>'
        for p in demanda.pendencias
    )
    return f"""
    <div style="margin-top:8px;">
        <div style="font-size:12px; color:#9ca3af; margin-bottom:6px;">
            Pendências
        </div>
        <div style="display:flex; flex-wrap:wrap; gap:4px;">{items}</div>
    </div>
    """


# ────────────────────────────────────────────────────────────
# LÓGICA DO CHAT
# ────────────────────────────────────────────────────────────

def responder(mensagem, historico, sessao_state, modo_str):
    if not mensagem.strip():
        return historico, sessao_state, renderizar_painel(None), gr.update()

    sessao = SessionState.model_validate(sessao_state) if sessao_state else nova_sessao()

    modo_map = {
        "GPU Local (Qwen3-4B)":   ModoExecucao.GPU_LOCAL,
        "CPU Local (Qwen3-1.7B)": ModoExecucao.CPU_LOCAL,
        "OpenAI (gpt-4o-mini)":   ModoExecucao.OPENAI,
    }
    sessao.modo_execucao = modo_map.get(modo_str, ModoExecucao.GPU_LOCAL)

    sessao, resposta, briefing = processar_turno(agente, sessao, mensagem)

    historico = historico or []
    historico.append({"role": "user",      "content": mensagem})
    historico.append({"role": "assistant", "content": resposta})

    painel_html = renderizar_painel(sessao)
    btn_update  = gr.update(visible=briefing is not None)

    return historico, sessao.model_dump(mode="json"), painel_html, btn_update


def aprovar_briefing(sessao_state):
    if not sessao_state:
        return None, "Nenhum briefing para aprovar."

    sessao  = SessionState.model_validate(sessao_state)
    demanda = sessao.demanda_ativa

    from schemas.models import BriefingOutput
    briefing = BriefingOutput.from_demand_state(demanda, sessao.modo_execucao)
    dados = briefing.to_sharepoint_dict()
    dados["_metadata"] = {
        "id_demanda":   briefing.id_demanda,
        "sessao_id":    briefing.sessao_id,
        "completude":   briefing.completude,
        "gerado_em":    briefing.gerado_em.isoformat(),
        "modo_execucao":briefing.modo_execucao.value,
        "proveniencia": {
            k: {"origem": v.origem.value, "turno": v.turno}
            for k, v in briefing.proveniencia.items()
        },
    }

    caminho = f"/tmp/briefing_{briefing.id_demanda}.json"
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(dados, f, ensure_ascii=False, indent=2)

    return caminho, "✅ Briefing aprovado!"


def reiniciar(historico, sessao_state):
    return [], {}, _painel_vazio(), gr.update(visible=False), ""


# ────────────────────────────────────────────────────────────
# CONSTRUÇÃO DA INTERFACE
# ────────────────────────────────────────────────────────────

def construir_interface(agente_compilado) -> gr.Blocks:
    global agente
    agente = agente_compilado

    with gr.Blocks(title="DataBrief AI") as demo:

        sessao_state = gr.State({})

        # ── Cabeçalho ────────────────────────────────────────
        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown(
                    "# 📋 DataBrief AI\n"
                    "*Agente conversacional para refinamento de demandas de dados*"
                )
            with gr.Column(scale=1, min_width=220):
                modo_selector = gr.Dropdown(
                    choices=[
                        "GPU Local (Qwen3-4B)",
                        "CPU Local (Qwen3-1.7B)",
                        "OpenAI (gpt-4o-mini)",
                    ],
                    value="GPU Local (Qwen3-4B)",
                    label="Modo de execução",
                    container=True,
                )

        gr.HTML("<hr style='margin:8px 0; border:none; border-top:1px solid #e5e7eb;'>")

        # ── Split screen ─────────────────────────────────────
        with gr.Row(equal_height=True):

            # Esquerda — Chat
            with gr.Column(scale=1):
                gr.Markdown("### 💬 Conversa")

                chatbot = gr.Chatbot(
                    value=[],
                    height=480,
                    show_label=False,
                    bubble_full_width=False,
                    type="messages",
                    placeholder=(
                        "**Olá! Sou o DataBrief AI.**\n\n"
                        "Descreva sua demanda de dados e vou ajudar a "
                        "estruturá-la em um briefing completo.\n\n"
                        "*Exemplo: \"Preciso de um dashboard de matrículas por polo\"*"
                    ),
                )

                msg_input = gr.Textbox(
                    placeholder="Descreva sua demanda...",
                    show_label=False,
                    lines=1,
                    max_lines=4,
                    submit_btn=True,
                )

                btn_reiniciar = gr.Button("🔄 Nova demanda", size="sm", variant="secondary")

            # Direita — Briefing
            with gr.Column(scale=1):
                gr.Markdown("### 📋 Briefing em construção")

                painel_briefing = gr.HTML(value=_painel_vazio())

                btn_aprovar = gr.Button(
                    "✅ Aprovar e baixar briefing",
                    visible=False,
                    variant="primary",
                    size="lg",
                )

                arquivo_download = gr.File(
                    label="Download do briefing",
                    visible=False,
                    file_types=[".json"],
                )

                status_aprovacao = gr.Markdown("")

        # ── Eventos ──────────────────────────────────────────
        msg_input.submit(
            fn=responder,
            inputs=[msg_input, chatbot, sessao_state, modo_selector],
            outputs=[chatbot, sessao_state, painel_briefing, btn_aprovar],
        ).then(fn=lambda: "", outputs=msg_input)

        btn_aprovar.click(
            fn=aprovar_briefing,
            inputs=[sessao_state],
            outputs=[arquivo_download, status_aprovacao],
        ).then(fn=lambda: gr.update(visible=True), outputs=arquivo_download)

        btn_reiniciar.click(
            fn=reiniciar,
            inputs=[chatbot, sessao_state],
            outputs=[chatbot, sessao_state, painel_briefing,
                     btn_aprovar, status_aprovacao],
        )

    return demo
