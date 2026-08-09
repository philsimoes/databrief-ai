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
    erro: Optional[str]


# ────────────────────────────────────────────────────────────
# PROMPTS
# ────────────────────────────────────────────────────────────

PROMPT_EXTRAIR = """Você é um assistente especializado em refinamento de demandas de dados.

Analise a resposta do usuário e extraia campos do briefing em JSON.
Retorne APENAS o JSON, sem explicações, sem markdown.

Contexto importante:
- A última pergunta feita pelo agente foi: "{ultima_pergunta}"
- A resposta do usuário deve ser interpretada como resposta a essa pergunta.
  Use isso para associar o conteúdo ao campo correto.
- Se a resposta for curta (1-3 palavras), interprete-a como resposta direta
  ao campo que estava sendo perguntado. Ex: se a pergunta era sobre formato
  de entrega e o usuário respondeu "Um dashboard", extraia resultado_esperado = "dashboard interativo".

Regras por tipo de campo:
- titulo, bloqueios, link_evidencia:
  extraia APENAS se explicitamente mencionado. Não invente.
- objetivo: extraia ou infira do contexto geral da demanda
- resultado_esperado: extraia APENAS se o usuário descreveu o que será entregue
- tipo_demanda: infira se possível. Valores: "Análise", "Estruturante", "Produto de Dados", "Alarmística"
- valor_negocio: infira pelo contexto. Valores: "Operacional", "Tático", "Estratégico"
  Dica: decisões de médio prazo ou investimento = Tático; rotina diária = Operacional; direção do negócio = Estratégico
- classificacao_estrategica: infira pelo contexto. Lista com um ou mais de:
  ["Priorização", "Insight para Decisão", "Estruturante", "Eficiência Operacional",
   "Monitoramento", "Qualidade de Dados", "Evolução de Produto", "Disponibilização de Informação"]
- perguntas_de_negocio: extraia APENAS se explicitamente mencionado

Texto do usuário:
{texto}

Estado atual (campos já preenchidos — não repita, não sobrescreva):
{estado_atual}

JSON:"""

PROMPT_PERGUNTA = """Você é um assistente especializado em refinamento de demandas de dados.

Campo prioritário a preencher agora: {campo_prioritario}
Outros campos ainda vazios: {outros_campos}
Contexto da demanda até agora: {contexto}

Regras:
- Faça UMA pergunta apenas sobre o campo prioritário
- Seja direto e específico, sem enrolação
- Use linguagem profissional mas natural
- Não mencione nomes de campos técnicos — reformule em linguagem humana

Instruções especiais por campo:
- Se campo for "valor_negocio": pergunte se o impacto é operacional (rotina diária), tático (decisões de médio prazo) ou estratégico (direcionamento do negócio)
- Se campo for "classificacao_estrategica": apresente as opções e peça para escolher uma ou mais: Priorização, Insight para Decisão, Estruturante, Eficiência Operacional, Monitoramento, Qualidade de Dados, Evolução de Produto, Disponibilização de Informação
- Se campo for "resultado_esperado": pergunte qual será o formato de entrega técnica.
  PROIBIDO mencionar qualquer palavra que o usuário usou (ex: "relatório", "análise", "dashboard").
  Use apenas estas opções padronizadas: dashboard interativo, agente automatizado,
  pipeline de dados, tabela Gold, modelo analítico.
  Formule assim: "Como será entregue — um dashboard interativo, um agente automatizado
  ou outro formato técnico?"
- Se campo for "tipo_demanda": pergunte se é uma Análise, Estruturante (pipeline/engenharia de dados), Produto de Dados (dashboard/agente) ou Alarmística (monitoramento com alertas)

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


def aplicar_extracao(demanda: DemandState, dados: dict, turno: int) -> DemandState:
    """Aplica os campos extraídos pelo Qwen ao estado da demanda."""

    def fp(valor):
        return FieldProvenance(valor=valor, origem=OrigemCampo.TEXT, turno=turno)

    if dados.get("titulo") and not demanda.titulo:
        demanda.titulo = fp(dados["titulo"])
    if dados.get("objetivo") and not demanda.objetivo:
        demanda.objetivo = fp(dados["objetivo"])
    if dados.get("resultado_esperado") and not demanda.resultado_esperado:
        demanda.resultado_esperado = fp(dados["resultado_esperado"])
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
        demanda.classificacao_estrategica = classificacoes

    if dados.get("perguntas_de_negocio") and not demanda.perguntas_de_negocio:
        demanda.perguntas_de_negocio = [
            fp(p) for p in dados["perguntas_de_negocio"] if p
        ]

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
    estado_atual = estado_para_texto(demanda)

    # Recupera a última resposta do agente para dar contexto ao Qwen.
    # Sem isso, o modelo não sabe a qual campo a resposta do usuário se refere.
    ultima_pergunta = state.get("ultima_resposta_agente", "") or ""

    prompt = PROMPT_EXTRAIR.format(
        texto=ultimo_turno.conteudo,
        estado_atual=estado_atual,
        ultima_pergunta=ultima_pergunta,
    )

    t0 = time.time()
    resposta_raw, _ = chamar_qwen(prompt, max_tokens=300)
    latencia = time.time() - t0

    dados = extrair_json(resposta_raw)
    demanda = aplicar_extracao(demanda, dados, demanda.turno_atual)
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
    precisa_estruturante = (
        demanda.tipo_demanda in DEPENDE_SEMPRE_DE_ESTRUTURANTE
        or (demanda.tipo_demanda == TipoDemanda.ANALISE
            and analise_depende_de_gold(demanda))
    )

    if precisa_estruturante:
        tem_estruturante = any(
            d.tipo_demanda == TipoDemanda.ESTRUTURANTE
            for d in sessao.demandas
        )
        if not tem_estruturante:
            nova = DemandState(
                tipo_demanda=TipoDemanda.ESTRUTURANTE,
                sessao_id=sessao.sessao_id
            )
            demanda.demandas_derivadas_ids.append(nova.id_demanda)
            sessao.demandas.append(nova)
            sessao.demandas = ordenar_demandas(sessao.demandas)
            sessao.indice_ativo = 0
            state["sessao"] = sessao
            state["ultima_resposta_agente"] = (
                "Percebi que essa demanda depende de uma tabela Gold que ainda "
                "não existe. Vou abrir uma demanda Estruturante primeiro. "
                "Me conta: qual tabela Gold precisamos construir?"
            )
            return state

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
    vazios = demanda.pendencias

    # Separa campo prioritário dos demais
    campo_prioritario = vazios[0] if vazios else ""
    outros_campos = ", ".join(vazios[1:]) if len(vazios) > 1 else "nenhum"

    prompt = PROMPT_PERGUNTA.format(
        campo_prioritario=campo_prioritario,
        outros_campos=outros_campos,
        contexto=contexto
    )

    t0 = time.time()
    pergunta, _ = chamar_qwen(prompt, max_tokens=150)
    latencia = time.time() - t0

    demanda.log_latencias[f"pergunta_turno_{demanda.turno_atual}"] = latencia
    sessao.demandas[sessao.indice_ativo] = demanda
    state["sessao"] = sessao
    state["ultima_resposta_agente"] = pergunta.strip()
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
    state["ultima_resposta_agente"] = (
        f"Briefing gerado! Aqui está o resumo:\n\n{resumo}\n\n"
        f"Completude: {briefing.completude:.0%}\n"
        f"Aguardando sua aprovação para download."
    )
    state["aguardando_aprovacao"] = True
    return state


# ────────────────────────────────────────────────────────────
# ROTEADOR
# ────────────────────────────────────────────────────────────

def rotear_apos_avaliacao(state: GraphState) -> str:
    sessao = state["sessao"]
    demanda = sessao.demanda_ativa
    if demanda.readiness == ReadinessStatus.PRONTA:
        return "gerar_briefing"
    return "formular_pergunta"


# ────────────────────────────────────────────────────────────
# CONSTRUÇÃO DO GRAFO
# ────────────────────────────────────────────────────────────

def construir_grafo():
    grafo = StateGraph(GraphState)
    grafo.add_node("extrair_campos",     no_extrair_campos)
    grafo.add_node("avaliar_completude", no_avaliar_completude)
    grafo.add_node("formular_pergunta",  no_formular_pergunta)
    grafo.add_node("gerar_briefing",     no_gerar_briefing)

    grafo.add_edge("extrair_campos",    "avaliar_completude")
    grafo.add_edge("formular_pergunta", END)
    grafo.add_edge("gerar_briefing",    END)

    grafo.add_conditional_edges(
        "avaliar_completude",
        rotear_apos_avaliacao,
        {
            "formular_pergunta": "formular_pergunta",
            "gerar_briefing":    "gerar_briefing",
        }
    )

    grafo.set_entry_point("extrair_campos")
    return grafo.compile()


# ────────────────────────────────────────────────────────────
# FUNÇÃO DE PROCESSAMENTO DE TURNO
# Exportada para uso direto no notebook e no Gradio
# ────────────────────────────────────────────────────────────

def processar_turno(agente, sessao: SessionState, texto_usuario: str) -> tuple:
    """
    Processa um turno completo: registra input, roda o grafo, retorna resposta.
    Retorna: (sessao_atualizada, resposta_agente, briefing_ou_None)

    A última pergunta feita pelo agente é preservada em sessao.ultima_pergunta_agente
    e passada no estado do grafo para que no_extrair_campos possa contextualizá-la.
    """
    demanda = sessao.demanda_ativa
    turno = TurnInput(conteudo=texto_usuario, tipo=TipoInput.TEXT)
    demanda.registrar_turno(turno)
    sessao.demandas[sessao.indice_ativo] = demanda

    # Recupera a última pergunta do agente (turno anterior) para passar como contexto
    ultima_pergunta_anterior = getattr(sessao, "ultima_pergunta_agente", "") or ""

    state = {
        "sessao": sessao,
        "ultima_resposta_agente": ultima_pergunta_anterior,
        "briefing_gerado": None,
        "aguardando_aprovacao": False,
        "erro": None,
    }

    resultado = agente.invoke(state)

    # Persiste a resposta do agente para uso no próximo turno
    sessao_resultado = resultado["sessao"]
    sessao_resultado.ultima_pergunta_agente = resultado["ultima_resposta_agente"]

    return (
        sessao_resultado,
        resultado["ultima_resposta_agente"],
        resultado["briefing_gerado"],
    )
