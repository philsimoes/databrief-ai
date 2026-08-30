# ============================================================
# graph/agent.py
# DataBrief AI — Bloco 02: Grafo LangGraph
# Módulo autossuficiente — não depende de variáveis do notebook
# ============================================================

# Correção de compatibilidade langchain/langchain-core
import langchain
if not hasattr(langchain, 'debug'):
    langchain.debug = False

import json, os, re, time
from typing import TypedDict, Optional, List
from langgraph.graph import StateGraph, END
from schemas.models import (
    DemandState, SessionState, TurnInput, FieldProvenance,
    TipoDemanda, OrigemCampo, TipoInput, ReadinessStatus,
    ModoExecucao, BriefingOutput, ValorNegocio, ClassificacaoEstrategica,
    DEPENDE_SEMPRE_DE_ESTRUTURANTE, analise_depende_de_gold, ordenar_demandas
)

# ────────────────────────────────────────────────────────────
# REFERÊNCIAS GLOBAIS AO MODELO E AO MODO DE EXECUÇÃO ATIVO
# Preenchidas por inicializar_modelo() antes de usar o grafo.
#
# _modo_execucao_ativo é a fonte única de verdade sobre qual modo está
# REALMENTE carregado nesta sessão do Colab (decidido pela variável
# MODO_EXECUCAO na Célula 1/2). A interface Gradio só exibe esse valor —
# não deixa mais escolher/trocar o modo por um dropdown editável, porque
# isso nunca trocava o modelo de verdade (só um metadado solto), o que é
# exatamente o tipo de "fallback silencioso" que o projeto proíbe.
# ────────────────────────────────────────────────────────────

_model = None
_tokenizer = None
_modo_execucao_ativo: Optional[ModoExecucao] = None

_ROTULOS_MODO = {
    ModoExecucao.GPU_LOCAL: "GPU Local (Qwen3-4B)",
    ModoExecucao.CPU_LOCAL: "CPU Local (Qwen3-1.7B)",
    ModoExecucao.OPENAI:    "OpenAI (gpt-4o-mini)",
}


def inicializar_modelo(model, tokenizer, modo: ModoExecucao):
    """
    Registra o modo de execução ativo desta sessão e, para os modos que usam
    um Qwen local (GPU_LOCAL, CPU_LOCAL), o model/tokenizer carregados no
    notebook. Chamar uma vez na Célula 2, depois de carregar o que for
    preciso para o modo escolhido em MODO_EXECUCAO.

    No modo OPENAI não existe Qwen local — a geração de texto vai inteira
    para a API (gpt-4o-mini). Nesse caso, chame
    inicializar_modelo(None, None, ModoExecucao.OPENAI).

    `modo` NÃO tem valor default de propósito (achado ao vivo, 30/08, Bloco
    15): antes tinha `= ModoExecucao.GPU_LOCAL`, o que significa que uma
    chamada na Célula 3 que por qualquer motivo deixasse de passar o 3º
    argumento (célula desatualizada, erro de cópia, cache do Colab) cairia
    SILENCIOSAMENTE em GPU_LOCAL — exatamente o tipo de fallback silencioso
    que este projeto proíbe explicitamente desde o Ato 1, e o oposto do que
    esta própria docstring promete ("nunca alterado silenciosamente"). Sem
    default, uma chamada que esqueça `modo` agora estoura um TypeError alto
    e claro na hora, em vez de rodar quieta no modo errado.
    """
    global _model, _tokenizer, _modo_execucao_ativo

    if not isinstance(modo, ModoExecucao):
        raise TypeError(
            f"inicializar_modelo() exige um ModoExecucao explícito em `modo` — "
            f"recebeu {modo!r}. Confirme que a Célula 3 passa "
            f"ModoExecucao(MODO_EXECUCAO) como terceiro argumento."
        )

    if modo != ModoExecucao.OPENAI and (model is None or tokenizer is None):
        raise RuntimeError(
            f"Modo {modo.value} exige model e tokenizer carregados na Célula 2 "
            "— receberam None. Verifique se o carregamento do Qwen rodou antes "
            "desta chamada."
        )

    _model = model
    _tokenizer = tokenizer
    _modo_execucao_ativo = modo
    print(f"✅ Modo ativo registrado: {_ROTULOS_MODO[modo]}")


def obter_modo_ativo() -> ModoExecucao:
    """
    Devolve o modo de execução realmente ativo nesta sessão (definido por
    inicializar_modelo() na Célula 2). Usado pela interface para exibir o
    modo ao usuário e por audio/transcricao.py para escolher o tamanho do
    Whisper — nunca inferido/adivinhado, sempre a mesma fonte única.
    """
    if _modo_execucao_ativo is None:
        raise RuntimeError(
            "Modo de execução não inicializado. Chame inicializar_modelo(model, "
            "tokenizer, modo) na Célula 2 antes de usar o grafo ou a interface."
        )
    return _modo_execucao_ativo


# ────────────────────────────────────────────────────────────
# CHAMADA AO MODELO
# ────────────────────────────────────────────────────────────

def chamar_qwen(prompt: str, max_tokens: int = 512) -> tuple:
    """Chama o Qwen local (GPU_LOCAL ou CPU_LOCAL) com um prompt e retorna
    (resposta, latência). Funciona igual nos dois modos — a diferença entre
    eles (tamanho do modelo, quantização 4-bit vs fp16/bf16 puro, cuda vs
    cpu) é decidida inteiramente na Célula 2, ao carregar model/tokenizer;
    aqui só se usa o que foi registrado, seja lá o que for."""
    import torch

    if _model is None or _tokenizer is None:
        raise RuntimeError(
            "Modelo local não inicializado. Chame inicializar_modelo(model, "
            "tokenizer, modo) primeiro — ou, se o modo ativo é OPENAI, use "
            "chamar_llm() em vez de chamar_qwen() diretamente."
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


def chamar_openai(prompt: str, max_tokens: int = 512) -> tuple:
    """Chama gpt-4o-mini via API da OpenAI e retorna (resposta, latência).

    Nunca cai silenciosamente para o Qwen se a chave não estiver configurada
    — levanta erro claro para o usuário corrigir a configuração, em vez de
    trocar de modelo por trás sem avisar (requisito do projeto: "nunca
    fallback silencioso")."""
    chave = os.environ.get("OPENAI_API_KEY")
    if not chave:
        raise RuntimeError(
            "Modo OPENAI está ativo, mas a variável de ambiente OPENAI_API_KEY "
            "não está configurada nesta sessão do Colab. Defina a chave (ex.: "
            "via Colab Secrets, ou os.environ['OPENAI_API_KEY'] = '...' na "
            "Célula 1) antes de carregar este modo — o sistema não usa o Qwen "
            "nem nenhum outro modelo no lugar sem avisar."
        )

    from openai import OpenAI
    cliente = OpenAI(api_key=chave)

    t0 = time.time()
    resposta = cliente.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.1,
    )
    latencia = time.time() - t0

    texto = (resposta.choices[0].message.content or "").strip()
    return texto, latencia


def chamar_llm(prompt: str, max_tokens: int = 512) -> tuple:
    """Ponto único de chamada ao modelo de linguagem usado por todos os nós
    do grafo. Despacha para o Qwen local (GPU_LOCAL/CPU_LOCAL) ou para a API
    da OpenAI (OPENAI) conforme o modo registrado em inicializar_modelo() —
    nunca por um valor solto vindo da interface. Se o modo ativo não tiver o
    que precisa (modelo local não carregado, chave da OpenAI ausente), a
    função chamada internamente levanta um erro claro em vez de usar outro
    modelo no lugar."""
    modo = obter_modo_ativo()
    if modo == ModoExecucao.OPENAI:
        return chamar_openai(prompt, max_tokens=max_tokens)
    return chamar_qwen(prompt, max_tokens=max_tokens)


# ────────────────────────────────────────────────────────────
# ESTADO DO GRAFO
# ────────────────────────────────────────────────────────────

class GraphState(TypedDict):
    sessao: SessionState
    ultima_resposta_agente: str
    briefing_gerado: Optional[dict]
    aguardando_aprovacao: bool
    campo_prioritario_atual: str   # campo sendo perguntado — usado pela interface para mostrar componentes especiais
    sugestao_pergunta_negocio: Optional[str]  # preenchido só quando campo_prioritario_atual
                                               # == "perguntas_de_negocio" — uma ou mais perguntas
                                               # candidatas geradas pelo Qwen a partir do objetivo
                                               # (uma por linha), mostradas num campo editável
                                               # para o usuário confirmar/editar/completar
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
- titulo, bloqueios, link_evidencia: Só se mencionado explicitamente. Se não houver
  menção, NÃO inclua a chave no JSON — nunca escreva frases como "não há bloqueios
  mencionados" ou "não mencionado" como se fosse um valor.
- classificacao_estrategica: Lista. Valores permitidos:
  "Priorização", "Insight para Decisão", "Estruturante", "Eficiência Operacional",
  "Monitoramento", "Qualidade de Dados", "Evolução de Produto", "Disponibilização de Informação"
- perguntas_de_negocio: NÃO extraia esse campo. Ele é tratado à parte, sempre
  sugerido pelo agente para confirmação do usuário — nunca preenchido direto
  a partir do texto livre.

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
    "titulo": (
        "Como você chamaria essa demanda? Pode ser um nome curto e descritivo."
    ),
}
# Nota: "perguntas_de_negocio" NÃO tem pergunta fixa — esse campo não é mais
# perguntado em aberto no chat. Ver PROMPT_SUGERIR_PERGUNTA e o tratamento
# especial em no_formular_pergunta: o agente sempre sugere uma pergunta
# candidata (gerada a partir do objetivo) para o usuário confirmar/editar,
# em vez de perguntar sempre a mesma coisa em texto livre.

PROMPT_SUGERIR_PERGUNTA = """Você é um assistente especializado em refinamento de demandas de dados.

Com base no objetivo da demanda abaixo, sugira pergunta(s) de negócio diretas
e específicas que essa demanda deveria responder — o tipo de pergunta que uma
análise, dashboard ou relatório deveria conseguir responder para quem pediu.

Prefira fortemente sugerir só UMA pergunta. Só sugira mais de uma (no máximo
3) se o objetivo mencionar explicitamente mais de um tema ou métrica
claramente diferentes. NUNCA sugira várias reformulações da mesma pergunta —
isso é um erro grave. Exemplo do que NÃO fazer (são a mesma pergunta com
palavras diferentes):
"Qual canal teve o maior crescimento de performance?"
"Qual canal teve a melhor taxa de conversão?"
"Qual canal teve o melhor ROI?"

Objetivo da demanda: {objetivo}
Contexto adicional: {contexto}

Regras:
- No máximo 3 perguntas, uma por linha, cada uma terminando em "?".
- Sem numeração, marcadores, explicações ou aspas — só as perguntas, uma por linha.
- Cada pergunta precisa investigar um aspecto do negócio genuinamente
  diferente das outras — nunca repita a mesma ideia com outras palavras.

Perguntas sugeridas:"""

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


_PADROES_SEM_RESPOSTA = (
    "não há", "nao ha", "não foi mencionado", "nao foi mencionado",
    "nenhum bloqueio", "nenhuma menção", "não mencionado", "nao mencionado",
    "n/a", "não aplicável", "nao aplicavel", "não se aplica", "nao se aplica",
    "não informado", "nao informado", "sem bloqueios", "sem bloqueio",
    "não identificado", "nao identificado",
)


def _normalizar_texto_campo(valor) -> str:
    """Bug real visto ao vivo (Ato 3, Bloco 12, Qwen3-1.7B/CPU_LOCAL): pra
    campos que deveriam ser um texto único (bloqueios, link_evidencia), o
    Qwen às vezes devolve uma LISTA em vez de string — observado quando o
    usuário menciona bloqueio E link no mesmo turno (ex.: ["depende da
    liberação de acesso..."] em vez de "depende da liberação de acesso...").
    Sem essa normalização, _parece_resposta_vazia() quebra com
    AttributeError: 'list' object has no attribute 'strip' — um crash real,
    não um fallback silencioso, mas ainda assim quebra o turno inteiro.
    Junta os itens da lista num texto só; qualquer outro tipo inesperado
    (dict, número) vira string via str() — nunca deixa passar adiante algo
    que não seja str.
    """
    if isinstance(valor, list):
        return "; ".join(str(v) for v in valor if v)
    if valor is None:
        return ""
    if isinstance(valor, str):
        return valor
    return str(valor)


def _parece_resposta_vazia(texto: str) -> bool:
    """Heurística leve: detecta quando o Qwen preenche um campo opcional
    (bloqueios, link_evidencia) com uma frase de "não há nada aqui" em vez de
    simplesmente omitir a chave — violação observada da instrução do prompt
    ("só se mencionado explicitamente"). Sem essa checagem, filler do tipo
    "não há bloqueios mencionados" entra no briefing como se fosse um dado
    real extraído do usuário/anexo, com badge de proveniência e tudo.
    """
    # str(...) aqui é rede de segurança extra — a normalização real acontece
    # em aplicar_extracao() via _normalizar_texto_campo(), antes de chegar
    # aqui, mas essa função pode ganhar outros chamadores no futuro.
    t = str(texto or "").strip().lower()
    if not t:
        return True
    return any(t.startswith(p) or p in t for p in _PADROES_SEM_RESPOSTA)


def aplicar_extracao(demanda: DemandState, dados: dict, turno: int, ultima_pergunta: str = "", turno_conteudo: str = "", origem: OrigemCampo = OrigemCampo.TEXT, nome_arquivo: str = None) -> DemandState:
    """Aplica os campos extraídos pelo Qwen ao estado da demanda.

    `origem` identifica de onde veio o turno que originou essa extração
    (TEXT digitado, AUDIO transcrito, ATTACHMENT de arquivo anexado) —
    propagada para o painel de proveniência via FieldProvenance.
    `nome_arquivo` só é relevante quando origem == ATTACHMENT — propagado ao
    campo FieldProvenance.arquivo para rastrear de qual anexo veio o dado.
    """

    def fp(valor):
        return FieldProvenance(valor=valor, origem=origem, turno=turno, arquivo=nome_arquivo)

    if dados.get("titulo") and not demanda.titulo:
        # Mesma classe de bug já vista em resultado_esperado: o prompt pede
        # "só se mencionado explicitamente", mas o Qwen3-4B às vezes preenche
        # título mesmo assim, geralmente reformulando o objetivo (bug real
        # observado: usuário nunca deu nome à demanda, e o campo veio
        # preenchido com uma paráfrase do objetivo). Como título é texto
        # livre, não dá pra validar contra uma lista fixa de palavras-chave
        # como fizemos em resultado_esperado — em vez disso, só aceita
        # quando a ÚLTIMA PERGUNTA DO AGENTE era de fato sobre título
        # (ver PERGUNTAS_FIXAS["titulo"]: "Como você chamaria essa
        # demanda?"). Isso garante que o valor só entra quando o usuário
        # estava realmente respondendo sobre título, não quando o Qwen
        # inventou por conta própria em qualquer outro turno.
        PALAVRAS_PERGUNTA_TITULO = ("chamaria", "nome", "título", "titulo")
        if any(p in ultima_pergunta.lower() for p in PALAVRAS_PERGUNTA_TITULO):
            demanda.titulo = fp(dados["titulo"])
    if dados.get("objetivo") and not demanda.objetivo:
        demanda.objetivo = fp(dados["objetivo"])
    if dados.get("resultado_esperado") and not demanda.resultado_esperado:
        # Só aceita resultado_esperado se o texto ORIGINAL do usuário (não a
        # extração do Qwen) contiver uma palavra-chave de formato explícita.
        # O prompt já instrui "extraia APENAS se o usuário disse
        # explicitamente dashboard/agente/tabela/pipeline", mas o Qwen3-4B
        # não obedece essa instrução com confiabilidade — mesma classe de
        # falha já vista em bloqueios/link_evidencia/perguntas_de_negocio.
        # Bug real observado: usuário descreveu só o tema da demanda
        # ("acompanhar a evolução da taxa de evasão... apresentar
        # mensalmente"), sem citar formato nenhum, e o Qwen preencheu
        # resultado_esperado como "Dashboard" mesmo assim — pulando
        # silenciosamente a pergunta de Radio que deveria ter aparecido.
        # A checagem antiga confiava em qualquer string que o Qwen
        # devolvesse desde que "parecesse" um formato válido, sem nunca
        # conferir contra o texto real do usuário; agora a fonte da verdade
        # é sempre turno_conteudo, em qualquer turno (não só o primeiro).
        PALAVRAS_FORMATO_EXPLICITO = {
            "dashboard": "Dashboard interativo",
            "painel": "Dashboard interativo",
            "consolidar": "Dashboard interativo",
            "pipeline": "Pipeline de dados",
            "tabela gold": "Tabela Gold",
            "modelo analítico": "Modelo analítico",
            "agente automatizado": "Agente automatizado",
            "agente": "Agente automatizado",
            "chatbot": "Agente automatizado",
            "conversar": "Agente automatizado",
            "converse": "Agente automatizado",
            "relatório automatizado": "Agente automatizado",
            "e-mail": "Agente automatizado",
            "alerta": "Agente automatizado",
        }
        texto_lower = turno_conteudo.lower()
        for chave, formato in PALAVRAS_FORMATO_EXPLICITO.items():
            if chave in texto_lower:
                demanda.resultado_esperado = fp(formato)
                break
    if dados.get("bloqueios") and not demanda.bloqueios:
        bloqueios_txt = _normalizar_texto_campo(dados["bloqueios"])
        if not _parece_resposta_vazia(bloqueios_txt):
            demanda.bloqueios = fp(bloqueios_txt)
    if dados.get("link_evidencia") and not demanda.link_evidencia:
        link_txt = _normalizar_texto_campo(dados["link_evidencia"])
        if not _parece_resposta_vazia(link_txt):
            demanda.link_evidencia = fp(link_txt)

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

    # perguntas_de_negocio NÃO é preenchido aqui. Bug real observado: o Qwen
    # preenchia esse campo direto a partir do texto livre (turno 1, sem
    # nenhuma pergunta ter sido feita sobre isso) — no caso do Phil, com o
    # próprio texto do objetivo copiado, nem formatado como pergunta. Isso
    # burlava silenciosamente a UI de sugestão/confirmação
    # (processar_confirmacao_pergunta_negocio) que é o único caminho
    # pretendido pra esse campo desde a mudança "sempre sugerida" — ver
    # PROMPT_SUGERIR_PERGUNTA e no_formular_pergunta. Extração livre e UI de
    # confirmação não podem coexistir pro mesmo campo: se a extração livre
    # preenche primeiro, a UI nunca aparece e o usuário nunca aprova nada.

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
    if ultimo_turno.conteudo in ("__checkbox__", "__radio__", "__sugestao_pergunta__"):
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
    resposta_raw, _ = chamar_llm(prompt, max_tokens=300)
    latencia = time.time() - t0

    dados = extrair_json(resposta_raw)
    demanda = aplicar_extracao(
        demanda, dados, demanda.turno_atual,
        ultima_pergunta=ultima_pergunta,
        turno_conteudo=ultimo_turno.conteudo,
        origem=origem,
        nome_arquivo=getattr(ultimo_turno, "nome_arquivo", None),
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

    # perguntas_de_negocio não é mais perguntado em aberto no chat (ver nota em
    # PERGUNTAS_FIXAS acima). Em vez de perguntar, o agente sempre gera UMA
    # pergunta de negócio candidata a partir do objetivo já informado e devolve
    # em sugestao_pergunta_negocio — a interface mostra num campo editável para
    # o usuário confirmar ou corrigir, em vez de repetir sempre a mesma pergunta
    # fixa em texto livre (decisão tomada em conversa — ver ato2_decisoes.md).
    if campo_prioritario == "perguntas_de_negocio":
        objetivo_txt = demanda.objetivo.valor if demanda.objetivo else "não informado"
        prompt_sugestao = PROMPT_SUGERIR_PERGUNTA.format(
            objetivo=objetivo_txt,
            contexto=contexto,
        )
        t0 = time.time()
        sugestao_raw, _ = chamar_llm(prompt_sugestao, max_tokens=200)
        latencia = time.time() - t0

        # Duas defesas contra o Qwen3-4B não seguir bem a instrução do prompt
        # (falha real observada em teste: pedido "normalmente só uma", mas o
        # modelo devolveu 7 reformulações da mesma pergunta, e a última linha
        # veio cortada pelo limite de tokens, sem "?", tipo "...maior variação
        # de"). Mesmo padrão de duas camadas já usado em bloqueios/link_evidencia:
        #
        # 1) Só aceita linha que termina em "?" — checagem estrita, pensada
        #    especificamente pra filtrar a própria geração do Qwen (que pode
        #    vir cortada pelo limite de tokens), não o texto digitado pelo
        #    usuário. Uma linha cortada/incompleta não pode vazar pro
        #    usuário como se fosse uma pergunta pronta.
        linhas = [l.strip().strip('"').strip("'").strip() for l in sugestao_raw.splitlines()]
        perguntas_sugeridas = [l for l in linhas if l.endswith("?")]
        if not perguntas_sugeridas:
            # Nada terminou em "?" — melhor mostrar a resposta bruta do Qwen
            # pro usuário editar do que devolver um campo vazio
            bruta = sugestao_raw.strip().strip('"').strip("'").strip()
            perguntas_sugeridas = [bruta] if bruta else []

        # 2) Limite duro de 3 — não confia só no prompt para conter o modelo
        perguntas_sugeridas = perguntas_sugeridas[:3]

        sugestao_texto = "\n".join(perguntas_sugeridas)
        mais_de_uma = len(perguntas_sugeridas) > 1

        demanda.log_latencias[f"pergunta_turno_{demanda.turno_atual}"] = latencia
        sessao.demandas[sessao.indice_ativo] = demanda
        state["sessao"] = sessao
        state["ultima_resposta_agente"] = (
            "Com base no que você já me contou, sugiro essas perguntas de negócio "
            "para essa demanda — edite se quiser (uma por linha) e confirme abaixo:"
            if mais_de_uma else
            "Com base no que você já me contou, sugiro essa pergunta de negócio "
            "para essa demanda — edite se quiser e confirme abaixo:"
        )
        state["campo_prioritario_atual"] = campo_prioritario
        state["sugestao_pergunta_negocio"] = sugestao_texto
        return state

    outros_campos = ", ".join(vazios[1:]) if len(vazios) > 1 else "nenhum"

    prompt = PROMPT_PERGUNTA.format(
        campo_prioritario=campo_prioritario,
        outros_campos=outros_campos,
        contexto=contexto
    )

    # Usa pergunta fixa se disponível — mais confiável que o Qwen para campos com opções conhecidas
    if campo_prioritario in PERGUNTAS_FIXAS:
        pergunta = PERGUNTAS_FIXAS[campo_prioritario]
        latencia = 0.0
    else:
        t0 = time.time()
        pergunta, _ = chamar_llm(prompt, max_tokens=150)
        latencia = time.time() - t0

    demanda.log_latencias[f"pergunta_turno_{demanda.turno_atual}"] = latencia
    sessao.demandas[sessao.indice_ativo] = demanda
    state["sessao"] = sessao
    state["ultima_resposta_agente"] = pergunta.strip()
    state["campo_prioritario_atual"] = campo_prioritario
    state["sugestao_pergunta_negocio"] = None
    return state


def _extrair_resumo_curto(texto_completo: str) -> str:
    """
    Extrai só o resumo executivo em linguagem natural do texto que o Qwen
    gera em no_gerar_briefing — PROMPT_BRIEFING pede resumo (2-3 frases) E
    a lista de campos formatada num único texto ("Campos Preenchidos:
    **Tipo de Demanda:** ... **Objetivo:** ..."). Essa lista já aparece
    visualmente no card do briefing — não faz sentido o TTS "ler" uma lista
    burocrática de rótulo/valor em voz alta (fica longo e cheio de markdown).
    Corta tudo a partir de "Campos Preenchidos" e mantém só o texto anterior.
    Se o marcador não aparecer (Qwen formatou diferente), devolve o texto
    inteiro — melhor ler tudo do que não ler nada.
    """
    match = re.search(r'campos\s+preenchidos', texto_completo, re.IGNORECASE)
    if match:
        return texto_completo[:match.start()].strip()
    return texto_completo.strip()


def no_gerar_briefing(state: GraphState) -> GraphState:
    sessao = state["sessao"]
    demanda = sessao.demanda_ativa

    contexto = estado_para_texto(demanda)
    prompt = PROMPT_BRIEFING.format(estado=contexto)

    t0 = time.time()
    resumo, _ = chamar_llm(prompt, max_tokens=400)
    latencia = time.time() - t0

    demanda.log_latencias[f"briefing_turno_{demanda.turno_atual}"] = latencia
    # Guarda só a parte de resumo em linguagem natural (sem a lista de campos
    # que vem em seguida, nem o texto de completude/aprovação em volta) — o
    # usuário já vê e implicitamente revisa esse texto antes de aprovar, e
    # ele é reaproveitado depois pelo TTS (Piper) sem precisar gerar de novo.
    demanda.resumo_executivo = _extrair_resumo_curto(resumo)
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

def processar_turno(agente, sessao: SessionState, texto_usuario: str, tipo: TipoInput = TipoInput.TEXT, nome_arquivo: str = None) -> tuple:
    """
    Processa um turno completo: registra input, roda o grafo, retorna resposta.
    Retorna: (sessao_atualizada, resposta_agente, briefing_ou_None,
              campo_prioritario_atual, sugestao_pergunta_negocio)

    `tipo` identifica se o turno veio de texto digitado, áudio transcrito ou
    arquivo anexado (TipoInput.TEXT / AUDIO / FILE) — propagado ao TurnInput
    e usado por no_extrair_campos para marcar a origem correta no painel
    de proveniência. `nome_arquivo` só é usado quando tipo == FILE — nome do
    anexo de onde veio o texto, propagado até o FieldProvenance.arquivo.

    A última pergunta feita pelo agente é preservada em sessao.ultima_pergunta_agente
    e passada no estado do grafo para que no_extrair_campos possa contextualizá-la.

    `sugestao_pergunta_negocio` só vem preenchida (str) quando
    campo_prioritario_atual == "perguntas_de_negocio" — é a pergunta candidata
    gerada pelo Qwen a partir do objetivo, para a interface mostrar num campo
    editável em vez de perguntar em aberto no chat. Nos demais casos vem None.
    """
    demanda = sessao.demanda_ativa
    turno = TurnInput(conteudo=texto_usuario, tipo=tipo, nome_arquivo=nome_arquivo)
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
        "sugestao_pergunta_negocio": None,
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
        resultado.get("sugestao_pergunta_negocio"),
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


def processar_confirmacao_pergunta_negocio(sessao: SessionState, texto_pergunta: str) -> SessionState:
    """
    Aplica a(s) pergunta(s) de negócio confirmada(s) (a sugestão gerada pelo
    Qwen em no_formular_pergunta, possivelmente editada/completada pelo
    usuário) direto ao estado. Chamada pela interface quando o usuário
    confirma o campo editável de sugestão — não passa pelo Qwen de novo, o
    texto já está pronto.

    Uma linha do campo editável = uma pergunta de negócio — o usuário pode
    apagar, editar ou adicionar linhas antes de confirmar. Linhas em branco
    são ignoradas.

    origem=MANUAL porque, mesmo a sugestão tendo sido gerada pelo sistema, o
    valor que efetivamente entra no briefing é sempre o que o usuário viu e
    confirmou (ou editou) — mesmo critério já usado em confirmar_resultado
    para resultado_esperado escolhido em componente visual.
    """
    demanda = sessao.demanda_ativa
    if not demanda or not texto_pergunta or not texto_pergunta.strip():
        return sessao

    perguntas = [l.strip() for l in texto_pergunta.splitlines() if l.strip()]
    if not perguntas:
        return sessao

    demanda.perguntas_de_negocio = [
        FieldProvenance(valor=p, origem=OrigemCampo.MANUAL, turno=demanda.turno_atual)
        for p in perguntas
    ]
    sessao.demandas[sessao.indice_ativo] = demanda
    return sessao
