# ============================================================
# graph/agent.py
# DataBrief AI — Bloco 02: Grafo LangGraph
# Módulo autossuficiente — não depende de variáveis do notebook
# ============================================================

# Correção de compatibilidade langchain/langchain-core
import langchain
if not hasattr(langchain, 'debug'):
    langchain.debug = False

import json, re, time
from typing import TypedDict, Optional, List
from langgraph.graph import StateGraph, END
from schemas.models import (
    DemandState, SessionState, TurnInput, FieldProvenance,
    TipoDemanda, OrigemCampo, TipoInput, ReadinessStatus,
    ModoExecucao, BriefingOutput, ValorNegocio, ClassificacaoEstrategica,
    DEPENDE_SEMPRE_DE_ESTRUTURANTE, analise_depende_de_gold, ordenar_demandas
)
from rag.contexto import buscar_contexto

# ────────────────────────────────────────────────────────────
# REFERÊNCIAS GLOBAIS AO MODELO
# Preenchidas por inicializar_modelo() antes de usar o grafo
# ────────────────────────────────────────────────────────────

_model = None
_tokenizer = None


def inicializar_modelo(model, tokenizer):
    """
    Recebe o model e tokenizer carregados no notebook e os
    armazena para uso nos nós do grafo.
    Chamar uma vez após carregar o Qwen na Célula 2.
    """
    global _model, _tokenizer
    _model = model
    _tokenizer = tokenizer
    print("✅ Modelo registrado no grafo")


# ────────────────────────────────────────────────────────────
# REFERÊNCIAS GLOBAIS AO RAG (Bloco 05b)
# Preenchidas por inicializar_rag() antes de usar o grafo.
# Se nunca forem inicializadas, o nó de busca de contexto simplesmente
# não encontra nada e o briefing segue sem citações — não derruba o app.
# ────────────────────────────────────────────────────────────

_corpus_rag = None
_indice_rag = None
_modelo_embeddings_rag = None


def inicializar_rag(corpus, indice, modelo_embeddings):
    """
    Recebe o corpus, o índice FAISS e o modelo de embeddings preparados
    pela célula de setup do notebook (rag.ingestao.preparar_corpus_e_indice)
    e os armazena para uso no nó de busca de contexto RAG.
    Chamar uma vez, depois de montar o Drive e preparar o corpus.
    """
    global _corpus_rag, _indice_rag, _modelo_embeddings_rag
    _corpus_rag = corpus
    _indice_rag = indice
    _modelo_embeddings_rag = modelo_embeddings
    print(f"✅ RAG registrado no grafo ({len(corpus)} chunks)")


# ────────────────────────────────────────────────────────────
# CHAMADA AO MODELO
# ────────────────────────────────────────────────────────────

def chamar_qwen(prompt: str, max_tokens: int = 512) -> tuple:
    """Chama o Qwen com um prompt e retorna (resposta, latência)."""
    import torch

    if _model is None or _tokenizer is None:
        raise RuntimeError(
            "Modelo não inicializado. Chame inicializar_modelo(model, tokenizer) primeiro."
        )

    mensagens = [{"role": "user", "content": prompt}]
    texto = _tokenizer.apply_chat_template(
        mensagens,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = _tokenizer(texto, return_tensors="pt").to(_model.device)

    t0 = time.time()
    with torch.no_grad():
        outputs = _model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.1,
            do_sample=True,
            pad_token_id=_tokenizer.eos_token_id,
        )
    latencia = time.time() - t0

    resposta = _tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )
    return resposta.strip(), latencia


# ────────────────────────────────────────────────────────────
# ESTADO DO GRAFO
# ────────────────────────────────────────────────────────────

class GraphState(TypedDict):
    sessao: SessionState
    ultima_resposta_agente: str
    briefing_gerado: Optional[dict]
    aguardando_aprovacao: bool
    campo_prioritario_atual: str   # campo sendo perguntado — usado pela interface para mostrar componentes especiais
    erro: Optional[str]


# ────────────────────────────────────────────────────────────
# PROMPTS
# ────────────────────────────────────────────────────────────

PROMPT_EXTRAIR = """Extraia campos de uma demanda de dados. Retorne APENAS JSON válido.

Pergunta anterior do agente: "{ultima_pergunta}"
Resposta do usuário: {texto}

Campos já preenchidos (não sobrescreva): {estado_atual}

Extraia apenas o que está explícito. Para cada campo, siga:
- objetivo: O que o usuário quer saber ou monitorar. Extraia sempre que possível.
- tipo_demanda: Use EXATAMENTE um destes valores ou null:
  "Produto de Dados" → dashboard, relatório automatizado, painel, agente
  "Análise" → análise pontual, investigação, entender algo
  "Estruturante" → pipeline, tabela Gold, engenharia de dados
  "Alarmística" → alerta, monitoramento com aviso, "sempre que", "quando X acontecer", receber notificação, receber e-mail, receber mensagem no Teams/Slack/WhatsApp/SMS, aviso automático, disparar alerta
  Na dúvida: null
- resultado_esperado: Formato técnico de entrega. Extraia APENAS se o usuário disse
  explicitamente "dashboard", "agente", "tabela", "pipeline". Não extraia se descreveu só o tema.
- valor_negocio: "Operacional", "Tático" ou "Estratégico". Infira pelo contexto.
- titulo, bloqueios, link_evidencia: Só se mencionado explicitamente.
- classificacao_estrategica: Lista. Valores permitidos:
  "Priorização", "Insight para Decisão", "Estruturante", "Eficiência Operacional",
  "Monitoramento", "Qualidade de Dados", "Evolução de Produto", "Disponibilização de Informação"
- perguntas_de_negocio: Só se mencionado explicitamente.

Se a pergunta anterior era sobre um campo específico e a resposta é curta, associe ao campo:
  formato de entrega → resultado_esperado
  impacto no negócio → valor_negocio
  título → titulo

JSON:"""

# Perguntas fixas por campo — usadas quando o campo tem opções conhecidas
# Evita que o Qwen 4-bit ignore o campo prioritário e invente perguntas
PERGUNTAS_FIXAS = {
    "valor_negocio": (
        "Essa demanda vai apoiar decisões do dia a dia, decisões de médio prazo "
        "ou o direcionamento estratégico do negócio?"
    ),
    "resultado_esperado": (
        "Como será entregue — um dashboard interativo, um agente automatizado "
        "ou outro formato técnico?"
    ),
    "tipo_demanda": (
        "Essa demanda é uma Análise pontual, um Produto de Dados (dashboard ou agente), "
        "uma Estruturante (pipeline/tabela Gold) ou uma Alarmística (monitoramento com alertas)?"
    ),
    "perguntas_de_negocio": (
        "Quais perguntas essa demanda precisa responder? "
        "Formule como perguntas diretas — por exemplo: 'Qual o índice de evasão por polo?' "
        "ou 'Quais tutores têm menor tempo de resposta?'"
    ),
    "perguntas_de_negocio_reformular": (
        "Preciso que sejam perguntas diretas, começando com 'Qual', 'Como', 'Quais' etc. "
        "Por exemplo: 'Qual o tempo médio de resposta do tutor?' — como ficaria a sua?"
    ),
    "titulo": (
        "Como você chamaria essa demanda? Pode ser um nome curto e descritivo."
    ),
}

PROMPT_PERGUNTA = """Você é um assistente especializado em refinamento de demandas de dados.

Campo prioritário a preencher agora: {campo_prioritario}
Contexto da demanda até agora: {contexto}

Faça UMA pergunta sobre o campo "{campo_prioritario}".
Use linguagem natural e profissional.
Não mencione nomes técnicos de campos.

Pergunta:"""

PROMPT_BRIEFING = """Você é um assistente especializado em refinamento de demandas de dados.

Com base no estado da demanda abaixo, gere um resumo executivo do briefing em 2-3 frases.
Depois liste os campos preenchidos de forma organizada.

Estado da demanda:
{estado}

Resumo e briefing formatado:"""


# ────────────────────────────────────────────────────────────
# FUNÇÕES AUXILIARES
# ────────────────────────────────────────────────────────────

def extrair_json(texto: str) -> dict:
    """Extrai JSON da resposta do modelo com fallback para regex."""
    texto = texto.strip()
    try:
        return json.loads(texto)
    except json.JSONDecodeError:
        pass
    match = re.search(r'\{.*\}', texto, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {}


def campos_vazios(demanda: DemandState) -> List[str]:
    """Retorna lista de campos obrigatórios ainda não preenchidos.

    Regra de inferência:
    - Para tipo Análise: resultado_esperado é inferível do objetivo.
      Se objetivo estiver preenchido, não pergunta resultado_esperado separadamente —
      será preenchido automaticamente em no_avaliar_completude.
    """
    vazios = []
    if not demanda.tipo_demanda:
        vazios.append("tipo_demanda")
    if not demanda.objetivo:
        vazios.append("objetivo")

    # resultado_esperado: obrigatório para todos os tipos, EXCETO Análise com objetivo preenchido
    if not demanda.resultado_esperado:
        analise_com_objetivo = (
            demanda.tipo_demanda == TipoDemanda.ANALISE
            and demanda.objetivo is not None
        )
        if not analise_com_objetivo:
            vazios.append("resultado_esperado")

    if not demanda.valor_negocio:
        vazios.append("valor_negocio")
    if not demanda.classificacao_estrategica:
        vazios.append("classificacao_estrategica")
    if not demanda.perguntas_de_negocio:
        vazios.append("perguntas_de_negocio")
    if not demanda.titulo:
        vazios.append("titulo")
    return vazios


def estado_para_texto(demanda: DemandState) -> str:
    """Converte o estado atual da demanda em texto para o prompt."""
    linhas = []
    if demanda.tipo_demanda:
        linhas.append(f"tipo_demanda: {demanda.tipo_demanda.value}")
    if demanda.objetivo:
        linhas.append(f"objetivo: {demanda.objetivo.valor}")
    if demanda.resultado_esperado:
        linhas.append(f"resultado_esperado: {demanda.resultado_esperado.valor}")
    if demanda.titulo:
        linhas.append(f"titulo: {demanda.titulo.valor}")
    if demanda.valor_negocio:
        linhas.append(f"valor_negocio: {demanda.valor_negocio.value}")
    if demanda.classificacao_estrategica:
        vals = [c.value for c in demanda.classificacao_estrategica]
        linhas.append(f"classificacao_estrategica: {vals}")
    if demanda.perguntas_de_negocio:
        perguntas = [fp.valor for fp in demanda.perguntas_de_negocio]
        linhas.append(f"perguntas_de_negocio: {perguntas}")
    return "\n".join(linhas) if linhas else "Nenhum campo preenchido ainda."


_INICIOS_PERGUNTA = (
    "qual", "quais", "quanto", "quantos", "quantas",
    "como", "quando", "onde", "quem", "por que", "porque", "pra que", "o que",
)


def _parece_pergunta_direta(texto: str) -> bool:
    """Heurística leve: o texto termina com '?' ou começa com uma palavra
    interrogativa comum. Usada como rede de segurança em aplicar_extracao —
    o Qwen3-4B nem sempre extrai perguntas_de_negocio de forma confiável
    mesmo quando o usuário já respondeu com uma pergunta direta válida
    (falha observada em testes: mesma resposta, ora extrai ora não).
    """
    t = (texto or "").strip().lower()
    if not t:
        return False
    if t.endswith("?"):
        return True
    return t.startswith(_INICIOS_PERGUNTA)


def aplicar_extracao(demanda: DemandState, dados: dict, turno: int, ultima_pergunta: str = "", turno_conteudo: str = "", origem: OrigemCampo = OrigemCampo.TEXT) -> DemandState:
    """Aplica os campos extraídos pelo Qwen ao estado da demanda.

    `origem` identifica de onde veio o turno que originou essa extração
    (TEXT digitado, AUDIO transcrito, ATTACHMENT de arquivo anexado) —
    propagada para o painel de proveniência via FieldProvenance.
    """

    def fp(valor):
        return FieldProvenance(valor=valor, origem=origem, turno=turno)

    if dados.get("titulo") and not demanda.titulo:
        demanda.titulo = fp(dados["titulo"])
    if dados.get("objetivo") and not demanda.objetivo:
        demanda.objetivo = fp(dados["objetivo"])
    if dados.get("resultado_esperado") and not demanda.resultado_esperado:
        # Aceita inferências óbvias de formato técnico — o usuário pode corrigir
        # no briefing final antes de aprovar (badge de proveniência mostra origem)
        FORMATOS_VALIDOS = {
            "dashboard", "dashboard interativo", "agente", "agente automatizado",
            "pipeline", "pipeline de dados", "tabela gold", "tabela", "modelo analítico",
            "relatório automatizado via agente", "outro"
        }
        # Mapeamento de inferências óbvias — contexto deixa o formato claro
        INFERENCIAS = {
            "converse": "agente automatizado",
            "conversar": "agente automatizado",
            "chatbot": "agente automatizado",
            "consolidar": "dashboard interativo",
            "painel": "dashboard interativo",
            "e-mail": "agente automatizado",
            "alerta": "agente automatizado",
        }
        valor_lower = str(dados["resultado_esperado"]).lower().strip()
        # Aceita se for formato válido direto
        if any(fmt in valor_lower for fmt in FORMATOS_VALIDOS):
            demanda.resultado_esperado = fp(dados["resultado_esperado"])
        # Ou tenta inferir pelo contexto da primeira mensagem
        elif not ultima_pergunta:  # só no primeiro turno, sem pergunta anterior
            texto_lower = turno_conteudo.lower()
            for chave, formato in INFERENCIAS.items():
                if chave in texto_lower:
                    demanda.resultado_esperado = fp(formato)
                    break
    if dados.get("bloqueios") and not demanda.bloqueios:
        demanda.bloqueios = fp(dados["bloqueios"])
    if dados.get("link_evidencia") and not demanda.link_evidencia:
        demanda.link_evidencia = fp(dados["link_evidencia"])

    if dados.get("tipo_demanda") and not demanda.tipo_demanda:
        try:
            demanda.tipo_demanda = TipoDemanda(dados["tipo_demanda"])
        except ValueError:
            pass

    if dados.get("valor_negocio") and not demanda.valor_negocio:
        try:
            demanda.valor_negocio = ValorNegocio(dados["valor_negocio"])
        except ValueError:
            pass

    if dados.get("classificacao_estrategica") and not demanda.classificacao_estrategica:
        classificacoes = []
        for c in dados["classificacao_estrategica"]:
            try:
                classificacoes.append(ClassificacaoEstrategica(c))
            except ValueError:
                pass
        # Limita a 3 classificações — mais que isso indica inferência imprecisa do Qwen
        demanda.classificacao_estrategica = classificacoes[:3]

    if dados.get("perguntas_de_negocio") and not demanda.perguntas_de_negocio:
        demanda.perguntas_de_negocio = [
            fp(p) for p in dados["perguntas_de_negocio"] if p
        ]
    # Se a última pergunta era sobre perguntas_de_negocio e o Qwen retornou lista
    # vazia, mas o texto do usuário já parece uma pergunta direta válida, usa o
    # texto bruto em vez de pedir reformulação à toa — evita o loop de "preciso
    # que sejam perguntas diretas" quando o usuário já respondeu corretamente
    # (a extração do Qwen3-4B falha esse campo com alguma frequência).
    elif (
        not demanda.perguntas_de_negocio
        and "pergunta" in ultima_pergunta.lower()
        and _parece_pergunta_direta(turno_conteudo)
    ):
        demanda.perguntas_de_negocio = [fp(turno_conteudo.strip())]

    return demanda


# ────────────────────────────────────────────────────────────
# NÓS DO GRAFO
# ────────────────────────────────────────────────────────────

def no_extrair_campos(state: GraphState) -> GraphState:
    sessao = state["sessao"]
    demanda = sessao.demanda_ativa

    if not demanda or not demanda.historico_turnos:
        return state

    ultimo_turno = demanda.historico_turnos[-1]

    # Mensagens internas dos componentes visuais — estado já foi atualizado, pula Qwen
    if ultimo_turno.conteudo in ("__checkbox__", "__radio__"):
        return state

    estado_atual = estado_para_texto(demanda)

    # Recupera a última resposta do agente para dar contexto ao Qwen.
    # Sem isso, o modelo não sabe a qual campo a resposta do usuário se refere.
    ultima_pergunta = state.get("ultima_resposta_agente", "") or ""

    # Mapeia o tipo do turno (TEXT/AUDIO/FILE) para a origem registrada
    # no painel de proveniência (TEXT/AUDIO/ATTACHMENT)
    MAPA_ORIGEM = {
        TipoInput.TEXT:  OrigemCampo.TEXT,
        TipoInput.AUDIO: OrigemCampo.AUDIO,
        TipoInput.FILE:  OrigemCampo.ATTACHMENT,
    }
    origem = MAPA_ORIGEM.get(ultimo_turno.tipo, OrigemCampo.TEXT)

    prompt = PROMPT_EXTRAIR.format(
        texto=ultimo_turno.conteudo,
        estado_atual=estado_atual,
        ultima_pergunta=ultima_pergunta,
    )

    t0 = time.time()
    resposta_raw, _ = chamar_qwen(prompt, max_tokens=300)
    latencia = time.time() - t0

    dados = extrair_json(resposta_raw)
    demanda = aplicar_extracao(
        demanda, dados, demanda.turno_atual,
        ultima_pergunta=ultima_pergunta,
        turno_conteudo=ultimo_turno.conteudo,
        origem=origem,
    )
    demanda.log_latencias[f"extracao_turno_{demanda.turno_atual}"] = latencia

    sessao.demandas[sessao.indice_ativo] = demanda
    state["sessao"] = sessao
    return state


def no_avaliar_completude(state: GraphState) -> GraphState:
    sessao = state["sessao"]
    demanda = sessao.demanda_ativa

    # Inferência automática: Análise com objetivo preenchido → resultado_esperado
    # Regra: para Análise, o resultado entregue é sempre uma análise/relatório derivado do objetivo.
    # Não faz sentido perguntar os dois separadamente.
    if (
        demanda.tipo_demanda == TipoDemanda.ANALISE
        and demanda.objetivo is not None
        and demanda.resultado_esperado is None
    ):
        valor_inferido = f"Análise: {demanda.objetivo.valor}"
        demanda.resultado_esperado = FieldProvenance(
            valor=valor_inferido,
            origem=OrigemCampo.RULE,
            turno=demanda.turno_atual,
        )

    # Verificar dependência de Estruturante
    # DESATIVADO no Ato 1 — fluxo multi-demanda será trabalhado no Ato 2
    # A lógica está preservada abaixo, comentada, para reativar no Ato 2
    # precisa_estruturante = (
    #     demanda.tipo_demanda in DEPENDE_SEMPRE_DE_ESTRUTURANTE
    #     or (demanda.tipo_demanda == TipoDemanda.ANALISE
    #         and analise_depende_de_gold(demanda))
    # )

    vazios = campos_vazios(demanda)
    if not vazios:
        demanda.readiness = ReadinessStatus.PRONTA
        demanda.pendencias = []
    else:
        demanda.readiness = ReadinessStatus.DISCOVERY
        demanda.pendencias = vazios

    sessao.demandas[sessao.indice_ativo] = demanda
    state["sessao"] = sessao
    return state


def no_formular_pergunta(state: GraphState) -> GraphState:
    sessao = state["sessao"]
    demanda = sessao.demanda_ativa

    contexto = estado_para_texto(demanda)
    # Recalcula campos_vazios diretamente — demanda.pendencias é zerado pelo LangGraph
    vazios = campos_vazios(demanda)

    # Separa campo prioritário dos demais
    campo_prioritario = vazios[0] if vazios else ""
    outros_campos = ", ".join(vazios[1:]) if len(vazios) > 1 else "nenhum"

    prompt = PROMPT_PERGUNTA.format(
        campo_prioritario=campo_prioritario,
        outros_campos=outros_campos,
        contexto=contexto
    )

    # Detecta se deve pedir reformulação de perguntas_de_negocio
    # Ocorre quando o campo prioritário é perguntas_de_negocio E a última pergunta
    # já era sobre esse campo — significa que o usuário respondeu sem formato de pergunta
    ultima_pergunta_sessao = getattr(sessao, "ultima_pergunta_agente", "") or ""
    pedir_reformulacao = (
        campo_prioritario == "perguntas_de_negocio"
        and "perguntas" in ultima_pergunta_sessao.lower()
        and not demanda.perguntas_de_negocio
    )
    chave_pergunta = "perguntas_de_negocio_reformular" if pedir_reformulacao else campo_prioritario

    # Usa pergunta fixa se disponível — mais confiável que o Qwen para campos com opções conhecidas
    if chave_pergunta in PERGUNTAS_FIXAS:
        pergunta = PERGUNTAS_FIXAS[chave_pergunta]
        latencia = 0.0
    else:
        t0 = time.time()
        pergunta, _ = chamar_qwen(prompt, max_tokens=150)
        latencia = time.time() - t0

    demanda.log_latencias[f"pergunta_turno_{demanda.turno_atual}"] = latencia
    sessao.demandas[sessao.indice_ativo] = demanda
    state["sessao"] = sessao
    state["ultima_resposta_agente"] = pergunta.strip()
    state["campo_prioritario_atual"] = campo_prioritario
    return state


def no_buscar_contexto_rag(state: GraphState) -> GraphState:
    """
    Busca contexto de mercado no corpus RAG a partir do objetivo da demanda.
    Roda uma única vez, quando a demanda fica pronta — não a cada turno,
    porque o objetivo pode mudar até lá. Falha do RAG nunca derruba o
    briefing: se não houver corpus carregado ou a busca falhar, o nó
    simplesmente segue sem citações.
    """
    sessao = state["sessao"]
    demanda = sessao.demanda_ativa

    if demanda.contexto_rag:
        return state  # já buscou pra essa demanda, não repete

    if _indice_rag is None or _corpus_rag is None or _modelo_embeddings_rag is None:
        return state  # RAG não inicializado — segue sem contexto

    if not demanda.objetivo:
        return state

    try:
        citacoes = buscar_contexto(
            demanda.objetivo.valor,
            _corpus_rag,
            _indice_rag,
            _modelo_embeddings_rag,
            chamar_qwen,
            k=5,
        )
    except Exception as e:
        print(f"⚠️ Busca de contexto RAG falhou: {e}")
        citacoes = []

    demanda.contexto_rag = [
        FieldProvenance(
            valor=c["citacao"],
            origem=OrigemCampo.RAG,
            turno=demanda.turno_atual,
            arquivo=c["fonte"],
            trecho_rag=c["citacao"],
        )
        for c in citacoes
    ]

    sessao.demandas[sessao.indice_ativo] = demanda
    state["sessao"] = sessao
    return state


def no_gerar_briefing(state: GraphState) -> GraphState:
    sessao = state["sessao"]
    demanda = sessao.demanda_ativa

    contexto = estado_para_texto(demanda)
    prompt = PROMPT_BRIEFING.format(estado=contexto)

    t0 = time.time()
    resumo, _ = chamar_qwen(prompt, max_tokens=400)
    latencia = time.time() - t0

    demanda.log_latencias[f"briefing_turno_{demanda.turno_atual}"] = latencia
    briefing = BriefingOutput.from_demand_state(demanda, sessao.modo_execucao)

    sessao.demandas[sessao.indice_ativo] = demanda
    state["sessao"] = sessao
    state["briefing_gerado"] = briefing.model_dump(mode="json")

    mensagem = (
        f"Briefing gerado! Aqui está o resumo:\n\n{resumo}\n\n"
        f"Completude: {briefing.completude:.0%}\n"
    )

    if briefing.fontes_rag:
        linhas_fontes = "\n".join(
            f"- {fp.arquivo} — {fp.valor}" for fp in briefing.fontes_rag
        )
        mensagem += f"\nContexto de mercado encontrado (RAG):\n{linhas_fontes}\n"

    mensagem += "\nAguardando sua aprovação para download."

    state["ultima_resposta_agente"] = mensagem
    state["aguardando_aprovacao"] = True
    return state


# ────────────────────────────────────────────────────────────
# ROTEADOR
# ────────────────────────────────────────────────────────────

def rotear_apos_avaliacao(state: GraphState) -> str:
    sessao = state["sessao"]
    demanda = sessao.demanda_ativa
    if demanda.readiness == ReadinessStatus.PRONTA:
        return "buscar_contexto_rag"
    return "formular_pergunta"


# ────────────────────────────────────────────────────────────
# CONSTRUÇÃO DO GRAFO
# ────────────────────────────────────────────────────────────

def construir_grafo():
    grafo = StateGraph(GraphState)
    grafo.add_node("extrair_campos",      no_extrair_campos)
    grafo.add_node("avaliar_completude",  no_avaliar_completude)
    grafo.add_node("formular_pergunta",   no_formular_pergunta)
    grafo.add_node("buscar_contexto_rag", no_buscar_contexto_rag)
    grafo.add_node("gerar_briefing",      no_gerar_briefing)

    grafo.add_edge("extrair_campos",      "avaliar_completude")
    grafo.add_edge("formular_pergunta",   END)
    grafo.add_edge("buscar_contexto_rag", "gerar_briefing")
    grafo.add_edge("gerar_briefing",      END)

    grafo.add_conditional_edges(
        "avaliar_completude",
        rotear_apos_avaliacao,
        {
            "formular_pergunta":   "formular_pergunta",
            "buscar_contexto_rag": "buscar_contexto_rag",
        }
    )

    grafo.set_entry_point("extrair_campos")
    return grafo.compile()


# ────────────────────────────────────────────────────────────
# FUNÇÃO DE PROCESSAMENTO DE TURNO
# Exportada para uso direto no notebook e no Gradio
# ────────────────────────────────────────────────────────────

def processar_turno(agente, sessao: SessionState, texto_usuario: str, tipo: TipoInput = TipoInput.TEXT) -> tuple:
    """
    Processa um turno completo: registra input, roda o grafo, retorna resposta.
    Retorna: (sessao_atualizada, resposta_agente, briefing_ou_None, campo_prioritario_atual)

    `tipo` identifica se o turno veio de texto digitado, áudio transcrito ou
    arquivo anexado (TipoInput.TEXT / AUDIO / FILE) — propagado ao TurnInput
    e usado por no_extrair_campos para marcar a origem correta no painel
    de proveniência.

    A última pergunta feita pelo agente é preservada em sessao.ultima_pergunta_agente
    e passada no estado do grafo para que no_extrair_campos possa contextualizá-la.
    """
    demanda = sessao.demanda_ativa
    turno = TurnInput(conteudo=texto_usuario, tipo=tipo)
    demanda.registrar_turno(turno)
    sessao.demandas[sessao.indice_ativo] = demanda

    # Recupera a última pergunta do agente (turno anterior) para passar como contexto
    ultima_pergunta_anterior = getattr(sessao, "ultima_pergunta_agente", "") or ""

    state = {
        "sessao": sessao,
        "ultima_resposta_agente": ultima_pergunta_anterior,
        "briefing_gerado": None,
        "aguardando_aprovacao": False,
        "campo_prioritario_atual": "",
        "erro": None,
    }

    resultado = agente.invoke(state)

    # Persiste a resposta do agente para uso no próximo turno
    sessao_resultado = resultado["sessao"]
    sessao_resultado.ultima_pergunta_agente = resultado["ultima_resposta_agente"]

    # Recalcula campos_vazios na saída — não confia em demanda.pendencias
    # porque o LangGraph zera listas durante serialização entre nós
    demanda_resultado = sessao_resultado.demanda_ativa
    campo_prioritario = ""
    if demanda_resultado and demanda_resultado.readiness != ReadinessStatus.PRONTA:
        vazios_agora = campos_vazios(demanda_resultado)
        campo_prioritario = vazios_agora[0] if vazios_agora else ""

    return (
        sessao_resultado,
        resultado["ultima_resposta_agente"],
        resultado["briefing_gerado"],
        campo_prioritario,
    )


def processar_selecao_checkbox(sessao: SessionState, selecoes: List[str]) -> SessionState:
    """
    Aplica a seleção múltipla de classificacao_estrategica direto ao estado.
    Chamada pela interface quando o usuário confirma o CheckboxGroup.
    Não passa pelo Qwen — as opções já vêm no formato correto.
    """
    demanda = sessao.demanda_ativa
    if not demanda or not selecoes:
        return sessao

    classificacoes = []
    for s in selecoes:
        try:
            classificacoes.append(ClassificacaoEstrategica(s))
        except ValueError:
            pass

    if classificacoes:
        demanda.classificacao_estrategica = classificacoes
        sessao.demandas[sessao.indice_ativo] = demanda

    return sessao
