# ============================================================
# interface/app.py
# DataBrief AI — Interface Gradio
# Chat centralizado + CheckboxGroup para classificacao_estrategica
# + entrada de áudio (microfone/upload) com transcrição revisável
# Gradio 5.x / 6.x compatível
# ============================================================

import gradio as gr
import json
from schemas.models import (
    SessionState, DemandState, ModoExecucao,
    ReadinessStatus, TipoDemanda, ClassificacaoEstrategica, TipoInput
)
from graph.agent import (
    processar_turno, processar_selecao_checkbox,
    processar_confirmacao_pergunta_negocio,
)
from audio.transcricao import transcrever_audio
from audio.sintese import sintetizar_texto
from attachments.extracao import extrair_texto_anexo

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
            LABEL_ORIGEM = {
                "TEXT":       "digitado",
                "AUDIO":      "áudio",
                "ATTACHMENT": "anexo",
                "RAG":        "base de conhecimento",
                "RULE":       "inferido pelo Qwen3-4B",
                "MANUAL":     "selecionado",
            }
            cor_origem = {
                "TEXT": "#3b82f6", "RULE": "#8b5cf6",
                "AUDIO": "#f59e0b", "RAG": "#22c55e",
                "ATTACHMENT": "#f97316", "MANUAL": "#6b7280",
            }.get(origem, "#9ca3af")
            label_origem = LABEL_ORIGEM.get(origem, origem.lower())
            turno_txt = f" · msg {turno}" if turno is not None else ""
            origem_badge = (
                f'<span style="font-size:10px;background:{cor_origem}18;color:{cor_origem};'
                f'border:1px solid {cor_origem}44;padding:1px 7px;border-radius:99px;'
                f'margin-left:6px;vertical-align:middle;">{label_origem}{turno_txt}</span>'
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


def processar_resposta(historico, sessao_state, modo_str, origem_mensagem="TEXT", nome_arquivo="", mensagem_override=""):
    # Passo 2: processa e adiciona a resposta do agente
    # origem_mensagem: "TEXT" (digitado), "AUDIO" (transcrição confirmada) ou
    # "FILE" (texto de anexo confirmado) — define o TipoInput do turno,
    # propagado até o painel de proveniência. nome_arquivo só é usado quando
    # origem_mensagem == "FILE".
    # mensagem_override: quando não vazio, é usado no lugar do texto que está
    # no chat — necessário para o anexo, onde o chat mostra só um rótulo
    # compacto (ex: "📎 arquivo.docx") mas o texto completo extraído precisa
    # ir inteiro pro pipeline de extração de campos.
    _vazio = (historico, sessao_state,
              _barra_html(0, "Aguardando demanda"), "",
              gr.update(visible=False), gr.update(visible=False),
              gr.update(visible=False), gr.update(visible=False),
              gr.update(value=[]), gr.update(visible=False), gr.update(value=None),
              gr.update(visible=False), gr.update(value=""))
    if not historico:
        return _vazio
    if mensagem_override and mensagem_override.strip():
        mensagem = mensagem_override
    else:
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

    MAPA_TIPO_TURNO = {"AUDIO": TipoInput.AUDIO, "FILE": TipoInput.FILE}
    tipo_turno = MAPA_TIPO_TURNO.get(origem_mensagem, TipoInput.TEXT)
    sessao, resposta, briefing, campo_atual, sugestao_pergunta = processar_turno(
        agente, sessao, mensagem, tipo=tipo_turno, nome_arquivo=nome_arquivo or None
    )
    historico.append({"role": "assistant", "content": resposta})

    barra_html    = renderizar_barra_progresso(sessao)
    pronto        = sessao.demanda_ativa and sessao.demanda_ativa.readiness == ReadinessStatus.PRONTA
    briefing_html = renderizar_briefing_final(sessao) if pronto else ""
    pede_classificacao = campo_atual == "classificacao_estrategica"
    pede_resultado     = campo_atual == "resultado_esperado"
    pede_pergunta_negocio = campo_atual == "perguntas_de_negocio"

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
        gr.update(visible=pede_pergunta_negocio),
        gr.update(value=sugestao_pergunta or ""),
    )


def confirmar_classificacao(selecoes, sessao_state, historico):
    """Chamada quando o usuário confirma a seleção do CheckboxGroup.
    Aplica a seleção e roda o grafo para formular a próxima pergunta.
    """
    if not selecoes or not sessao_state:
        # Mesma forma (10 valores) do retorno normal — o retorno anterior tinha
        # só 4 valores, descasado com os 10 outputs do evento (bug: se o
        # usuário clicasse "Confirmar seleção" sem marcar nada, o Gradio
        # recebia menos valores do que o esperado).
        return (historico, sessao_state,
                gr.update(visible=False), gr.update(visible=False),
                "", gr.update(visible=False), gr.update(visible=False),
                _barra_html(0, "Aguardando demanda"),
                gr.update(visible=False), gr.update(value=""))

    sessao = SessionState.model_validate(sessao_state)
    sessao = processar_selecao_checkbox(sessao, selecoes)

    # Registra a seleção no histórico como mensagem do usuário
    texto_confirmado = ", ".join(selecoes)
    historico = historico or []
    historico.append({"role": "user", "content": f"Classificação: {texto_confirmado}"})

    # Roda o grafo com uma mensagem vazia para avançar para a próxima pergunta
    # O grafo vai avaliar o estado atualizado e formular o próximo campo
    sessao, resposta, briefing, campo_atual, sugestao_pergunta = processar_turno(agente, sessao, "__checkbox__")

    historico.append({"role": "assistant", "content": resposta})

    pronto = sessao.demanda_ativa and sessao.demanda_ativa.readiness == ReadinessStatus.PRONTA
    briefing_html_val = renderizar_briefing_final(sessao) if pronto else ""
    pede_classificacao = campo_atual == "classificacao_estrategica"
    # classificacao_estrategica costuma ser seguida por perguntas_de_negocio na
    # ordem de campos_vazios — precisa mostrar a caixa de sugestão se for o caso
    pede_pergunta_negocio = campo_atual == "perguntas_de_negocio"

    return (
        historico,
        sessao.model_dump(mode="json"),
        gr.update(visible=False),               # esconde o checkbox
        gr.update(visible=pede_classificacao),  # mostra de novo só se cair no mesmo campo
        briefing_html_val,
        gr.update(visible=pronto),              # secao_briefing
        gr.update(visible=pronto),              # btn_aprovar
        renderizar_barra_progresso(sessao),     # barra atualizada
        gr.update(visible=pede_pergunta_negocio),
        gr.update(value=sugestao_pergunta or ""),
    )


def confirmar_resultado(selecao, sessao_state, historico):
    """Chamada quando o usuário confirma o Radio de resultado_esperado.
    Aplica o valor direto ao estado e avança o grafo.
    """
    if not selecao or not sessao_state:
        return (historico, sessao_state, gr.update(visible=False), "",
                gr.update(visible=False), gr.update(visible=False),
                _barra_html(0, "Aguardando demanda"),
                gr.update(visible=False), gr.update(value=""))

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

    sessao, resposta, briefing, campo_atual, sugestao_pergunta = processar_turno(agente, sessao, "__radio__")
    historico.append({"role": "assistant", "content": resposta})

    pronto = sessao.demanda_ativa and sessao.demanda_ativa.readiness == ReadinessStatus.PRONTA
    briefing_html_val = renderizar_briefing_final(sessao) if pronto else ""
    # resultado_esperado costuma ser seguido por classificacao_estrategica, mas
    # em alguns fluxos (ex.: campo inferido) pode cair direto em perguntas_de_negocio
    pede_pergunta_negocio = campo_atual == "perguntas_de_negocio"

    return (
        historico,
        sessao.model_dump(mode="json"),
        gr.update(visible=False),
        briefing_html_val,
        gr.update(visible=pronto),
        gr.update(visible=pronto),
        renderizar_barra_progresso(sessao),
        gr.update(visible=pede_pergunta_negocio),
        gr.update(value=sugestao_pergunta or ""),
    )


def confirmar_pergunta_negocio(texto, sessao_state, historico):
    """Chamada quando o usuário confirma (ou edita) a pergunta de negócio
    sugerida pelo agente. Aplica direto ao estado — não passa pelo Qwen de
    novo — e roda o grafo pra avançar pro próximo campo.
    """
    if not texto or not texto.strip() or not sessao_state:
        return (historico, sessao_state, gr.update(visible=False), "",
                gr.update(visible=False), gr.update(visible=False),
                _barra_html(0, "Aguardando demanda"))

    sessao = SessionState.model_validate(sessao_state)
    sessao = processar_confirmacao_pergunta_negocio(sessao, texto)

    historico = historico or []
    historico.append({"role": "user", "content": f"Pergunta de negócio: {texto.strip()}"})

    sessao, resposta, briefing, campo_atual, _ = processar_turno(agente, sessao, "__sugestao_pergunta__")
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
        return None, gr.update(visible=False), "Nenhum briefing para aprovar.", gr.update(visible=False)
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
            k: {"origem": v.origem.value, "turno": v.turno, "arquivo": v.arquivo}
            for k, v in briefing.proveniencia.items()
        },
    }
    caminho = f"/tmp/briefing_{briefing.id_demanda}.json"
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(dados, f, ensure_ascii=False, indent=2)
    # Botão "Ouvir resumo" (TTS) só aparece depois da aprovação explícita —
    # Piper nunca carrega antes disso (fluxo de aprovação humana do projeto).
    return caminho, gr.update(visible=True), "Briefing aprovado. Faça o download abaixo.", gr.update(visible=True)


def gerar_audio_resumo(sessao_state):
    """Chamada quando o usuário clica em "Ouvir resumo" — só disponível depois
    de aprovar o briefing. Sintetiza o resumo executivo (já gerado pelo Qwen
    e mostrado ao usuário antes da aprovação) com o Piper TTS, que roda em
    CPU e é carregado/liberado sob demanda, sem disputar VRAM com o Qwen.
    """
    if not sessao_state:
        return gr.update(visible=False), "Nenhum briefing aprovado para sintetizar.", sessao_state
    sessao  = SessionState.model_validate(sessao_state)
    demanda = sessao.demanda_ativa
    if not demanda or not demanda.resumo_executivo:
        return gr.update(visible=False), "Resumo não disponível para este briefing.", sessao_state

    try:
        caminho_wav = f"/tmp/resumo_{demanda.id_demanda}.wav"
        resultado = sintetizar_texto(demanda.resumo_executivo, caminho_wav)
    except Exception as e:
        return gr.update(visible=False), f"Não foi possível gerar o áudio: {e}", sessao_state

    demanda.log_latencias["tts"] = resultado["latencia_s"]
    sessao.demandas[sessao.indice_ativo] = demanda

    return gr.update(value=caminho_wav, visible=True), "", sessao.model_dump(mode="json")


def reiniciar(historico, sessao_state):
    sessao_limpa = nova_sessao()
    return (
        [], sessao_limpa.model_dump(mode="json"),
        _barra_html(0, "Aguardando demanda"), "",
        gr.update(visible=False), gr.update(visible=False),
        gr.update(visible=False), gr.update(visible=False),
        gr.update(value=[]), "",
        gr.update(visible=False), gr.update(value=None, visible=False), "",
        gr.update(visible=False), gr.update(value=""),
    )


# ────────────────────────────────────────────────────────────
# LÓGICA DE ÁUDIO
# ────────────────────────────────────────────────────────────

def transcrever_audio_ui(caminho_audio):
    """Chamada quando o usuário termina de gravar ou envia um arquivo de áudio.
    Transcreve (carrega Whisper, transcreve, libera a GPU) e mostra o texto
    num campo editável para revisão antes de entrar no chat.
    """
    if not caminho_audio:
        return gr.update(visible=False), ""
    resultado = transcrever_audio(caminho_audio)
    return gr.update(visible=True), resultado["texto"]


def enviar_transcricao(texto_transcrito, historico):
    """Chamada quando o usuário confirma a transcrição (já revisada/corrigida)
    e ela entra no chat como se fosse uma mensagem digitada.
    """
    if not texto_transcrito or not texto_transcrito.strip():
        return historico, "", gr.update(visible=False)
    historico = historico or []
    historico.append({"role": "user", "content": texto_transcrito})
    return historico, "", gr.update(visible=False)


# ────────────────────────────────────────────────────────────
# LÓGICA DE ANEXO — PDF/DOCX/TXT
# ────────────────────────────────────────────────────────────

def extrair_anexo_ui(caminho_arquivo):
    """Chamada quando o usuário anexa um documento (PDF, DOCX ou TXT).
    Extrai o texto e mostra num campo editável para revisão antes de entrar
    no chat — útil porque a extração pode trazer ruído de formatação (quebras
    de linha, cabeçalhos repetidos) que vale a pena limpar antes de enviar.
    """
    if not caminho_arquivo:
        return gr.update(visible=False), "", ""
    resultado = extrair_texto_anexo(caminho_arquivo)
    texto = resultado["texto"]
    if resultado["truncado"]:
        texto += (
            "\n\n[...texto truncado — o documento é muito longo; revise e "
            "edite antes de enviar se precisar de outro trecho...]"
        )
    return gr.update(visible=True), texto, resultado["arquivo"]


def enviar_anexo(texto_anexo, historico, nome_arquivo):
    """Chamada quando o usuário confirma o texto extraído do anexo (já
    revisado/editado). O chat mostra só um rótulo compacto do arquivo — não o
    texto inteiro, que pode ser bem longo — mas o texto completo é preservado
    à parte (texto_anexo_pendente_state) para alimentar o pipeline de
    extração de campos em processar_resposta.
    """
    if not texto_anexo or not texto_anexo.strip():
        return historico, "", gr.update(visible=False), ""
    historico = historico or []
    rotulo = f"📎 {nome_arquivo}" if nome_arquivo else "📎 documento anexado"
    historico.append({"role": "user", "content": rotulo})
    return historico, "", gr.update(visible=False), texto_anexo


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
        origem_mensagem_state = gr.State("TEXT")
        nome_arquivo_state = gr.State("")
        texto_anexo_pendente_state = gr.State("")

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

            # ── Sugestão de pergunta de negócio ───────────────
            # perguntas_de_negocio não é mais perguntado em aberto: o agente
            # sempre sugere uma pergunta candidata (gerada a partir do
            # objetivo) e o usuário confirma ou edita aqui antes de seguir.
            with gr.Column(visible=False) as secao_sugestao_pergunta:
                gr.Markdown("**Pergunta de negócio sugerida** — edite se quiser e confirme:")
                editor_pergunta_negocio = gr.Textbox(lines=2, show_label=False)
                btn_confirmar_pergunta_negocio = gr.Button(
                    "Confirmar pergunta", variant="primary", size="sm"
                )

            msg_input = gr.Textbox(
                placeholder="Descreva sua demanda...",
                show_label=False, lines=1, max_lines=4, submit_btn=True,
            )

            btn_reiniciar = gr.Button("Nova demanda", size="sm", variant="secondary")

            # ── Entrada de áudio — microfone/upload + transcrição revisável ──
            with gr.Row():
                audio_input = gr.Audio(
                    sources=["microphone", "upload"],
                    type="filepath",
                    label="Ou grave/envie um áudio (até 5 min)",
                )

            with gr.Column(visible=False) as secao_transcricao:
                gr.Markdown("**Revise a transcrição antes de enviar** (corrija siglas ou nomes se precisar):")
                editor_transcricao = gr.Textbox(lines=3, show_label=False)
                with gr.Row():
                    btn_confirmar_transcricao = gr.Button(
                        "Usar esta transcrição", variant="primary", size="sm"
                    )
                    btn_descartar_transcricao = gr.Button(
                        "Descartar", size="sm", variant="secondary"
                    )

            # ── Entrada de anexo — PDF/DOCX/TXT + texto revisável ─────
            with gr.Row():
                anexo_input = gr.File(
                    file_types=[".pdf", ".docx", ".txt"],
                    label="Ou anexe um documento (PDF, DOCX ou TXT)",
                    type="filepath",
                )

            with gr.Column(visible=False) as secao_anexo:
                gr.Markdown("**Revise o texto extraído antes de enviar** (limpe ruído de formatação se precisar):")
                editor_anexo = gr.Textbox(lines=5, show_label=False)
                with gr.Row():
                    btn_confirmar_anexo = gr.Button(
                        "Usar este texto", variant="primary", size="sm"
                    )
                    btn_descartar_anexo = gr.Button(
                        "Descartar", size="sm", variant="secondary"
                    )

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

            # ── TTS — só aparece depois da aprovação ──────────
            btn_ouvir_resumo = gr.Button(
                "🔊 Ouvir resumo", size="sm", variant="secondary", visible=False
            )
            audio_resumo = gr.Audio(
                label="Resumo em áudio", visible=False, autoplay=True
            )
            status_tts = gr.Markdown("")

        # ── Eventos ──────────────────────────────────────────
        msg_input.submit(
            fn=adicionar_mensagem_usuario,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, msg_input],
        ).then(
            fn=lambda: "TEXT",
            outputs=[origem_mensagem_state],
        ).then(
            fn=processar_resposta,
            inputs=[chatbot, sessao_state, modo_selector, origem_mensagem_state],
            outputs=[
                chatbot, sessao_state, barra_progresso, briefing_html,
                secao_briefing, btn_aprovar, arquivo_download,
                secao_checkbox, checkbox_classificacao,
                secao_radio_resultado, radio_resultado,
                secao_sugestao_pergunta, editor_pergunta_negocio,
            ],
        )

        audio_input.change(
            fn=transcrever_audio_ui,
            inputs=[audio_input],
            outputs=[secao_transcricao, editor_transcricao],
        )

        btn_confirmar_transcricao.click(
            fn=enviar_transcricao,
            inputs=[editor_transcricao, chatbot],
            outputs=[chatbot, editor_transcricao, secao_transcricao],
        ).then(
            fn=lambda: None,
            outputs=[audio_input],
        ).then(
            fn=lambda: "AUDIO",
            outputs=[origem_mensagem_state],
        ).then(
            fn=processar_resposta,
            inputs=[chatbot, sessao_state, modo_selector, origem_mensagem_state],
            outputs=[
                chatbot, sessao_state, barra_progresso, briefing_html,
                secao_briefing, btn_aprovar, arquivo_download,
                secao_checkbox, checkbox_classificacao,
                secao_radio_resultado, radio_resultado,
                secao_sugestao_pergunta, editor_pergunta_negocio,
            ],
        )

        btn_descartar_transcricao.click(
            fn=lambda: ("", gr.update(visible=False), None),
            outputs=[editor_transcricao, secao_transcricao, audio_input],
        )

        anexo_input.change(
            fn=extrair_anexo_ui,
            inputs=[anexo_input],
            outputs=[secao_anexo, editor_anexo, nome_arquivo_state],
        )

        btn_confirmar_anexo.click(
            fn=enviar_anexo,
            inputs=[editor_anexo, chatbot, nome_arquivo_state],
            outputs=[chatbot, editor_anexo, secao_anexo, texto_anexo_pendente_state],
        ).then(
            fn=lambda: None,
            outputs=[anexo_input],
        ).then(
            fn=lambda: "FILE",
            outputs=[origem_mensagem_state],
        ).then(
            fn=processar_resposta,
            inputs=[chatbot, sessao_state, modo_selector, origem_mensagem_state, nome_arquivo_state, texto_anexo_pendente_state],
            outputs=[
                chatbot, sessao_state, barra_progresso, briefing_html,
                secao_briefing, btn_aprovar, arquivo_download,
                secao_checkbox, checkbox_classificacao,
                secao_radio_resultado, radio_resultado,
                secao_sugestao_pergunta, editor_pergunta_negocio,
            ],
        )

        btn_descartar_anexo.click(
            fn=lambda: ("", gr.update(visible=False), None, ""),
            outputs=[editor_anexo, secao_anexo, anexo_input, nome_arquivo_state],
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
                secao_sugestao_pergunta, editor_pergunta_negocio,
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
                secao_sugestao_pergunta, editor_pergunta_negocio,
            ],
        )

        btn_confirmar_pergunta_negocio.click(
            fn=confirmar_pergunta_negocio,
            inputs=[editor_pergunta_negocio, sessao_state, chatbot],
            outputs=[
                chatbot, sessao_state,
                secao_sugestao_pergunta,
                briefing_html,
                secao_briefing, btn_aprovar,
                barra_progresso,
            ],
        )

        btn_aprovar.click(
            fn=aprovar_briefing,
            inputs=[sessao_state],
            outputs=[arquivo_download, arquivo_download, status_aprovacao, btn_ouvir_resumo],
        )

        btn_ouvir_resumo.click(
            fn=gerar_audio_resumo,
            inputs=[sessao_state],
            outputs=[audio_resumo, status_tts, sessao_state],
        )

        btn_reiniciar.click(
            fn=reiniciar,
            inputs=[chatbot, sessao_state],
            outputs=[
                chatbot, sessao_state, barra_progresso, briefing_html,
                secao_briefing, btn_aprovar, arquivo_download,
                secao_checkbox, checkbox_classificacao, status_aprovacao,
                btn_ouvir_resumo, audio_resumo, status_tts,
                secao_sugestao_pergunta, editor_pergunta_negocio,
            ],
        )

    return demo
