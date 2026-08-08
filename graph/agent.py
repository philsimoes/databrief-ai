# ============================================================
# Célula 4: Grafo LangGraph — graph/agent.py
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
    ModoExecucao, BriefingOutput
)

# ────────────────────────────────────────────────────────────
# ESTADO DO GRAFO
# TypedDict é o formato que o LangGraph usa para passar
# informações entre os nós.
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

Analise o texto abaixo e extraia APENAS os campos explicitamente mencionados pelo usuário.
NÃO infira, NÃO suponha, NÃO complete campos que não foram ditos.
Se um campo não foi mencionado, simplesmente não o inclua no JSON.
Retorne APENAS o JSON, sem explicações, sem markdown.

Campos possíveis:
- titulo: string curta descritiva (só se o usuário nomeou a demanda)
- tipo_demanda: um de ["Análise", "Estruturante", "Produto de Dados", "Alarmística"] (só se mencionado)
- objetivo: o que se quer alcançar (só se descrito)
- resultado_esperado: o que será entregue (só se descrito)
- valor_negocio: um de ["Operacional", "Tático", "Estratégico"] (só se mencionado)
- classificacao_estrategica: lista — só se o usuário mencionou explicitamente
- perguntas_de_negocio: lista de perguntas que a demanda responde (só se mencionadas)
- bloqueios: impedimentos (só se mencionados)
- link_evidencia: URL (só se fornecida)

Texto do usuário:
{texto}

Estado atual (campos já preenchidos — não repita, não sobrescreva):
{estado_atual}

JSON com APENAS o que foi explicitamente dito:"""

PROMPT_PERGUNTA = """Você é um assistente especializado em refinamento de demandas de dados.

Campos obrigatórios ainda vazios: {campos_vazios}
Contexto da demanda até agora: {contexto}

Regras:
- Faça UMA pergunta apenas, a mais importante para avançar
- Priorize: tipo_demanda > objetivo > resultado_esperado > valor_negocio > perguntas_de_negocio
- Seja direto e específico, sem enrolação
- Use linguagem profissional mas natural
- Não mencione nomes de campos técnicos como "tipo_demanda" — reformule em linguagem humana

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
    # Tenta parse direto
    try:
        return json.loads(texto)
    except json.JSONDecodeError:
        pass
    # Tenta extrair bloco JSON com regex
    match = re.search(r'\{.*\}', texto, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {}


def campos_vazios(demanda: DemandState) -> List[str]:
    """Retorna lista de campos obrigatórios ainda não preenchidos."""
    vazios = []
    if not demanda.tipo_demanda:
        vazios.append("tipo_demanda")
    if not demanda.objetivo:
        vazios.append("objetivo")
    if not demanda.resultado_esperado:
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
    return "\n".join(linhas) if linhas else "Nenhum campo preenchido ainda."


def aplicar_extracao(demanda: DemandState, dados: dict, turno: int) -> DemandState:
    """Aplica os campos extraídos pelo Qwen ao estado da demanda."""

    def fp(valor):
        return FieldProvenance(valor=valor, origem=OrigemCampo.TEXT, turno=turno)

    if "titulo" in dados and dados["titulo"] and not demanda.titulo:
        demanda.titulo = fp(dados["titulo"])

    if "objetivo" in dados and dados["objetivo"] and not demanda.objetivo:
        demanda.objetivo = fp(dados["objetivo"])

    if "resultado_esperado" in dados and dados["resultado_esperado"] and not demanda.resultado_esperado:
        demanda.resultado_esperado = fp(dados["resultado_esperado"])

    if "bloqueios" in dados and dados["bloqueios"] and not demanda.bloqueios:
        demanda.bloqueios = fp(dados["bloqueios"])

    if "link_evidencia" in dados and dados["link_evidencia"] and not demanda.link_evidencia:
        demanda.link_evidencia = fp(dados["link_evidencia"])

    if "tipo_demanda" in dados and not demanda.tipo_demanda:
        try:
            demanda.tipo_demanda = TipoDemanda(dados["tipo_demanda"])
        except ValueError:
            pass

    if "valor_negocio" in dados and not demanda.valor_negocio:
        from schemas.models import ValorNegocio
        try:
            demanda.valor_negocio = ValorNegocio(dados["valor_negocio"])
        except ValueError:
            pass

    if "classificacao_estrategica" in dados and not demanda.classificacao_estrategica:
        from schemas.models import ClassificacaoEstrategica
        classificacoes = []
        for c in (dados["classificacao_estrategica"] or []):
            try:
                classificacoes.append(ClassificacaoEstrategica(c))
            except ValueError:
                pass
        demanda.classificacao_estrategica = classificacoes

    if "perguntas_de_negocio" in dados and not demanda.perguntas_de_negocio:
        demanda.perguntas_de_negocio = [
            fp(p) for p in (dados["perguntas_de_negocio"] or []) if p
        ]

    return demanda


# ────────────────────────────────────────────────────────────
# NÓS DO GRAFO
# ────────────────────────────────────────────────────────────

def no_extrair_campos(state: GraphState) -> GraphState:
    """
    Nó 1: Extrai campos do último turno do usuário.
    Chama o Qwen com o texto e o estado atual.
    """
    sessao = state["sessao"]
    demanda = sessao.demanda_ativa

    if not demanda or not demanda.historico_turnos:
        return state

    ultimo_turno = demanda.historico_turnos[-1]
    estado_atual = estado_para_texto(demanda)

    prompt = PROMPT_EXTRAIR.format(
        texto=ultimo_turno.conteudo,
        estado_atual=estado_atual
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
    """
    Nó 2: Avalia completude e detecta dependências entre demandas.
    - Produto de Dados e Alarmística sempre precisam de Estruturante.
    - Análise precisa de Estruturante só se usar Gold como fonte.
    Se a dependência não existe na sessão, cria a demanda Estruturante
    automaticamente e reordena a fila.
    """
    from schemas.models import (
        DEPENDE_SEMPRE_DE_ESTRUTURANTE, analise_depende_de_gold,
        ordenar_demandas
    )

    sessao = state["sessao"]
    demanda = sessao.demanda_ativa

    # Verificar dependência de Estruturante
    precisa_estruturante = (
        demanda.tipo_demanda in DEPENDE_SEMPRE_DE_ESTRUTURANTE
        or (demanda.tipo_demanda == TipoDemanda.ANALISE
            and analise_depende_de_gold(demanda))
    )

    if precisa_estruturante:
        # Verificar se já existe Estruturante na sessão
        tem_estruturante = any(
            d.tipo_demanda == TipoDemanda.ESTRUTURANTE
            for d in sessao.demandas
        )
        if not tem_estruturante:
            # Criar demanda Estruturante automaticamente
            nova = DemandState(
                tipo_demanda=TipoDemanda.ESTRUTURANTE,
                sessao_id=sessao.sessao_id
            )
            demanda.demandas_derivadas_ids.append(nova.id_demanda)
            sessao.demandas.append(nova)
            sessao.demandas = ordenar_demandas(sessao.demandas)
            # Reposicionar no índice do Estruturante (sempre índice 0)
            sessao.indice_ativo = 0
            state["sessao"] = sessao
            state["ultima_resposta_agente"] = (
                "Percebi que essa demanda depende de uma tabela Gold que ainda "
                "não existe. Vou abrir uma demanda Estruturante primeiro. "
                "Me conta: qual tabela Gold precisamos construir?"
            )
            return state

    # Sem dependência pendente — avaliar campos normalmente
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
    """
    Nó 3: Formula a próxima pergunta para o usuário.
    Sempre uma pergunta por vez, priorizando campos bloqueantes.
    """
    sessao = state["sessao"]
    demanda = sessao.demanda_ativa

    contexto = estado_para_texto(demanda)
    vazios = demanda.pendencias

    prompt = PROMPT_PERGUNTA.format(
        campos_vazios=", ".join(vazios),
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
    """
    Nó 4: Gera o briefing final quando todos os campos estão preenchidos.
    Aguarda aprovação humana antes de liberar.
    """
    sessao = state["sessao"]
    demanda = sessao.demanda_ativa

    contexto = estado_para_texto(demanda)
    prompt = PROMPT_BRIEFING.format(estado=contexto)

    t0 = time.time()
    resumo, _ = chamar_qwen(prompt, max_tokens=400)
    latencia = time.time() - t0

    demanda.log_latencias[f"briefing_turno_{demanda.turno_atual}"] = latencia

    # Gera o BriefingOutput estruturado
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
# ROTEADOR — decide para onde ir após avaliar_completude
# ────────────────────────────────────────────────────────────

def rotear_apos_avaliacao(state: GraphState) -> str:
    """
    Decide o próximo nó com base no readiness da demanda ativa.
    Nunca faz fallback silencioso — sempre retorna um destino explícito.
    """
    sessao = state["sessao"]
    demanda = sessao.demanda_ativa

    if demanda.readiness == ReadinessStatus.PRONTA:
        return "gerar_briefing"
    else:
        return "formular_pergunta"


# ────────────────────────────────────────────────────────────
# CONSTRUÇÃO DO GRAFO
# ────────────────────────────────────────────────────────────

def construir_grafo() -> StateGraph:
    grafo = StateGraph(GraphState)

    # Adicionar nós
    grafo.add_node("extrair_campos",    no_extrair_campos)
    grafo.add_node("avaliar_completude", no_avaliar_completude)
    grafo.add_node("formular_pergunta", no_formular_pergunta)
    grafo.add_node("gerar_briefing",    no_gerar_briefing)

    # Adicionar arestas fixas
    grafo.add_edge("extrair_campos",    "avaliar_completude")
    grafo.add_edge("formular_pergunta", END)
    grafo.add_edge("gerar_briefing",    END)

    # Aresta condicional após avaliação
    grafo.add_conditional_edges(
        "avaliar_completude",
        rotear_apos_avaliacao,
        {
            "formular_pergunta": "formular_pergunta",
            "gerar_briefing":    "gerar_briefing",
        }
    )

    # Ponto de entrada
    grafo.set_entry_point("extrair_campos")

    return grafo.compile()


# Compilar o grafo
agente = construir_grafo()
print("✅ Grafo LangGraph compilado com sucesso")
print(f"   Nós: {list(agente.get_graph().nodes.keys())}")
