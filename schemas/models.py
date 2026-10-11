# ============================================================
# schemas/models.py
# DataBrief AI — Bloco 01: Schemas Pydantic (revisado no Bloco 30)
# ============================================================

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ────────────────────────────────────────────────────────────
# ENUMS — listas fechadas de valores permitidos
# ────────────────────────────────────────────────────────────

class TipoDemanda(str, Enum):
    """Tipo técnico da entrega.

    Bloco 23 — "Estruturante" foi removida (06/09, decisão do Phil como
    gerente de dados): na prática, pipeline/tabela Gold/engenharia de dados
    sempre serviam de base para um Produto de Dados ou uma Alarmística — não
    existia demanda de negócio pedindo só isso, sem um consumível final.
    Passam a entrar direto em PRODUTO_DADOS. A distinção "isso é trabalho
    estruturante" continua existindo, mas como tag em classificacao_estrategica
    (ClassificacaoEstrategica.ESTRUTURANTE, abaixo — campo diferente, não afetado
    por essa mudança), não como tipo_demanda separado.
    """
    ANALISE        = "Análise"
    PRODUTO_DADOS  = "Produto de Dados"
    ALARMASTICA    = "Alarmística"


class ClassificacaoEstrategica(str, Enum):
    """Natureza do valor gerado. Uma demanda pode ter múltiplas."""
    PRIORIZACAO              = "Priorização"
    INSIGHT_DECISAO          = "Insight para Decisão"
    ESTRUTURANTE             = "Estruturante"
    EFICIENCIA_OPERACIONAL   = "Eficiência Operacional"
    MONITORAMENTO            = "Monitoramento"
    QUALIDADE_DADOS          = "Qualidade de Dados"
    EVOLUCAO_PRODUTO         = "Evolução de Produto"
    DISPONIBILIZACAO_INFO    = "Disponibilização de Informação"


class ValorNegocio(str, Enum):
    """Nível de impacto estratégico da demanda."""
    OPERACIONAL  = "Operacional"
    TATICO       = "Tático"
    ESTRATEGICO  = "Estratégico"


class StatusDemanda(str, Enum):
    """Estado atual da demanda no ciclo de vida."""
    EM_ANDAMENTO = "Em andamento"
    CONCLUIDA    = "Concluída"
    BLOQUEADA    = "Bloqueada"
    PARADO       = "Parado"


class OrigemCampo(str, Enum):
    """De onde veio o valor de um campo — usado no painel de proveniência."""
    TEXT       = "TEXT"        # digitado pelo usuário
    AUDIO      = "AUDIO"       # transcrito pelo Whisper
    ATTACHMENT = "ATTACHMENT"  # extraído de arquivo anexado
    RAG        = "RAG"         # recuperado do corpus
    RULE       = "RULE"        # determinado por regra do sistema
    MANUAL     = "MANUAL"      # corrigido manualmente pelo usuário


class ReadinessStatus(str, Enum):
    """
    Estado de completude da demanda.
    O agente usa esse status para decidir se pode gerar o briefing.
    """
    PRONTA        = "pronta"        # todos os campos obrigatórios preenchidos
    ESCLARECIMENTO = "esclarecimento" # campos preenchidos mas precisam confirmação
    DISCOVERY     = "discovery"     # campos obrigatórios ainda em aberto
    BLOQUEADA     = "bloqueada"     # dependência externa impede avanço, OU
                                     # (Bloco 20) todos os campos obrigatórios
                                     # ainda vazios já estão em
                                     # campos_desistidos — o agente parou de
                                     # perguntar ativamente, precisa que o
                                     # usuário traga a informação sozinho


class ModoExecucao(str, Enum):
    """
    Modo de execução do sistema.
    Nunca alterado silenciosamente — sempre explícito na interface.
    """
    GPU_LOCAL = "GPU_LOCAL"   # Qwen3-4B 4-bit + Whisper Small
    CPU_LOCAL = "CPU_LOCAL"   # Qwen3-1.7B + Whisper Tiny
    OPENAI    = "OPENAI"      # gpt-4o-mini via API


class CamadaRAG(str, Enum):
    """
    Camada do corpus RAG a consultar.
    Existe desde o Ato 1 mesmo que ambas apontem para corpus fictício.
    """
    PUBLICA  = "PUBLICA"   # informações públicas YDUQS/setor
    INTERNA  = "INTERNA"   # catálogo interno, taxonomia, faróis
    AMBAS    = "AMBAS"


# ────────────────────────────────────────────────────────────
# DEPENDÊNCIAS ENTRE TIPOS DE DEMANDA
# Legado do Ato 1/2, DESATIVADO desde então (fluxo multi-demanda nunca foi
# ligado — ver comentário em no_avaliar_completude, graph/agent.py).
# Bloco 30 — removidos DEPENDE_SEMPRE_DE_ESTRUTURANTE, FONTES_GOLD e
# analise_depende_de_gold(): dependiam do tipo "Estruturante" (removido no
# Bloco 23) e de CamposAnalise.fonte_dados (schema morto, removido abaixo).
# Fica só a ordenação simples, porque SessionState.adicionar_demanda() ainda
# chama ordenar_demandas(). Retomar multi-demanda está no backlog.
# ────────────────────────────────────────────────────────────

ORDEM_RESOLUCAO: Dict[TipoDemanda, int] = {
    TipoDemanda.ANALISE:       1,
    TipoDemanda.PRODUTO_DADOS: 2,
    TipoDemanda.ALARMASTICA:   2,
}


def ordenar_demandas(demandas: List["DemandState"]) -> List["DemandState"]:
    return sorted(demandas, key=lambda d: ORDEM_RESOLUCAO.get(d.tipo_demanda, 99))

# ────────────────────────────────────────────────────────────
# FIELD PROVENANCE
# Rastreia a origem de cada campo individualmente.
# ────────────────────────────────────────────────────────────

class FieldProvenance(BaseModel):
    """
    Registra de onde veio o valor de um campo.
    Usado no painel de proveniência e nas métricas de F1 por campo.
    """
    valor: Any                          # valor atual do campo
    origem: OrigemCampo                 # TEXT / AUDIO / ATTACHMENT / RAG / RULE / MANUAL
    turno: int                          # número do turno em que foi preenchido
    arquivo: Optional[str] = None       # nome do arquivo se origem = ATTACHMENT ou RAG
    trecho_rag: Optional[str] = None    # trecho recuperado se origem = RAG
    confirmado: bool = False            # True após aprovação explícita do usuário
    timestamp: datetime = Field(        # quando o campo foi preenchido
        default_factory=datetime.now
    )
    timestamp_audio: Optional[float] = None  # início do segmento em segundos (só AUDIO)


# ────────────────────────────────────────────────────────────
# TURN INPUT
# Representa uma entrada do usuário em um turno.
# ────────────────────────────────────────────────────────────

class TipoInput(str, Enum):
    TEXT  = "text"
    AUDIO = "audio"
    FILE  = "file"


class TurnInput(BaseModel):
    """
    Representa uma entrada do usuário em um turno da conversa.
    O hash evita reprocessamento duplicado (especialmente para áudio).
    """
    conteudo: str                          # texto bruto ou transcrição
    tipo: TipoInput                        # text / audio / file
    timestamp: datetime = Field(default_factory=datetime.now)
    hash_conteudo: str = ""                # preenchido automaticamente no validator
    prompt_inicial_whisper: Optional[str] = None  # glossário de siglas para o ASR
    nome_arquivo: Optional[str] = None     # nome do anexo, quando tipo == FILE —
                                            # propagado ao FieldProvenance.arquivo

    @field_validator("hash_conteudo", mode="before")
    @classmethod
    def gerar_hash(cls, v: str, info: Any) -> str:
        """Gera hash SHA-256 do conteúdo para deduplicação."""
        conteudo = info.data.get("conteudo", "")
        return hashlib.sha256(conteudo.encode()).hexdigest()


# ────────────────────────────────────────────────────────────
# CAMPOS ESPECÍFICOS POR TIPO DE DEMANDA — removidos no Bloco 30
# CamposAnalise / CamposEstruturante / CamposProdutoDados / CamposAlarmastica
# existiam desde o Ato 1, mas o grafo nunca leu nem escreveu nenhum deles
# (achado do Ato 3). Readiness e completude sempre dependeram só dos 7 campos
# universais. Extração de campos por tipo (ex.: threshold de Alarmística)
# está no backlog — ver README, "Próximos passos".
# ────────────────────────────────────────────────────────────

# ────────────────────────────────────────────────────────────
# DEMAND STATE
# Estado completo de uma única demanda ao longo da conversa.
# ────────────────────────────────────────────────────────────

class DemandState(BaseModel):
    """
    Estado completo de uma demanda.
    Todos os campos rastreiam proveniência individual: os de texto livre via
    FieldProvenance direto; os 3 de opção fixa via proveniencia_campos_fixos.
    Atualizado a cada turno; nunca substituído, apenas acumulado.
    """
    # Identificação
    id_demanda: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    sessao_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])

    # Campos universais com proveniência
    titulo:                   Optional[FieldProvenance] = None
    tipo_demanda:             Optional[TipoDemanda]     = None
    classificacao_estrategica: List[ClassificacaoEstrategica] = Field(default_factory=list)
    valor_negocio:            Optional[ValorNegocio]    = None
    objetivo:                 Optional[FieldProvenance] = None
    perguntas_de_negocio:     List[FieldProvenance]     = Field(default_factory=list)
    resultado_esperado:       Optional[FieldProvenance] = None
    bloqueios:                Optional[FieldProvenance] = None
    link_evidencia:           Optional[FieldProvenance] = None
    status:                   StatusDemanda             = StatusDemanda.PARADO
    resumo_executivo:         Optional[str] = None   # gerado pelo Qwen em
                                                       # no_gerar_briefing — texto
                                                       # já mostrado e implicitamente
                                                       # revisado pelo usuário antes da
                                                       # aprovação; reaproveitado pelo
                                                       # TTS (Piper) após aprovar

    # Bloco 30 — proveniência dos 3 campos de opção fixa (tipo_demanda,
    # valor_negocio, classificacao_estrategica). Esses campos continuam
    # guardando o valor cru (enum/lista de enum) porque muita lógica compara
    # direto (ex.: demanda.tipo_demanda == TipoDemanda.ANALISE) — trocar o
    # tipo deles pra FieldProvenance mexeria em agent.py, app.py e nos dois
    # scripts de avaliação. Em vez disso, a origem fica registrada aqui ao
    # lado, chave = nome do campo. Escrita só via registrar_proveniencia_fixa().
    proveniencia_campos_fixos: Dict[str, FieldProvenance] = Field(default_factory=dict)

    # Estado do agente
    readiness:      ReadinessStatus = ReadinessStatus.DISCOVERY
    pendencias:     List[str]       = Field(default_factory=list)
    turno_atual:    int             = 0
    historico_turnos: List[TurnInput] = Field(default_factory=list)

    # Quantas vezes cada campo já foi perguntado sem ainda ter sido resolvido —
    # campo -> contagem. Usado só pra variar o texto quando a MESMA pergunta
    # está sendo repetida (ver _texto_retentativa em graph/agent.py), pra não
    # parecer que o agente travou repetindo a frase idêntica.
    tentativas_pergunta: Dict[str, int] = Field(default_factory=dict)

    # Campos que o agente desistiu de perguntar ativamente, depois de 3
    # tentativas seguidas sem conseguir extrair um valor (ver
    # no_avaliar_completude/no_formular_pergunta em graph/agent.py, Bloco 20).
    # Continuam contando como pendência de verdade (nunca somem de
    # campos_vazios()/pendencias, nunca deixam a demanda chegar em PRONTA
    # sozinhos) — só param de ser escolhidos como campo_prioritario_atual,
    # pra não repetir a mesma pergunta pra sempre. Se o usuário mencionar o
    # valor espontaneamente em qualquer turno futuro (sem ter sido
    # perguntado), a extração normal continua funcionando igual — "desistir"
    # é só sobre parar de insistir com a pergunta, não sobre esquecer o
    # campo.
    campos_desistidos: List[str] = Field(default_factory=list)

    # Demandas derivadas detectadas nesta sessão
    # Ex: Gold necessária antes de um Produto de Dados
    demandas_derivadas_ids: List[str] = Field(default_factory=list)

    # Métricas e log (obrigatório desde o Ato 1)
    log_latencias: Dict[str, float] = Field(default_factory=dict)
    # Chaves esperadas: "extracao", "pergunta", "briefing", "asr", "tts"

    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    def registrar_turno(self, turn_input: TurnInput) -> None:
        """Adiciona um turno ao histórico e incrementa o contador."""
        self.historico_turnos.append(turn_input)
        self.turno_atual += 1
        self.updated_at = datetime.now()

    def registrar_proveniencia_fixa(
        self, campo: str, valor: Any, origem: OrigemCampo,
        arquivo: Optional[str] = None,
    ) -> None:
        """Bloco 30 — registra de onde veio tipo_demanda / valor_negocio /
        classificacao_estrategica. Sempre chamada no mesmo lugar em que o
        valor cru do campo é gravado, pra os dois nunca divergirem."""
        self.proveniencia_campos_fixos[campo] = FieldProvenance(
            valor=valor, origem=origem, turno=self.turno_atual, arquivo=arquivo,
        )

    def campos_obrigatorios_preenchidos(self) -> bool:
        """
        Verifica se os campos universais obrigatórios estão presentes.
        """
        universais = [
            self.titulo,
            self.tipo_demanda,
            self.objetivo,
            self.resultado_esperado,
        ]
        return all(c is not None for c in universais)

    def calcular_completude(self) -> float:
        """
        Retorna fração de campos obrigatórios preenchidos (0.0 a 1.0).
        Usado na métrica de completude do briefing (meta >= 0.85).
        """
        campos = [
            self.titulo,
            self.tipo_demanda,
            self.objetivo,
            self.resultado_esperado,
            self.valor_negocio,
            len(self.classificacao_estrategica) > 0,
            len(self.perguntas_de_negocio) > 0,
        ]
        preenchidos = sum(1 for c in campos if c)
        return preenchidos / len(campos)


# ────────────────────────────────────────────────────────────
# SESSION STATE
# Estado completo de uma sessão com múltiplas demandas.
# Suporta demandas compostas com ordenação por dependência.
# ────────────────────────────────────────────────────────────

class SessionState(BaseModel):
    """
    Estado de uma sessão completa.
    Pode conter múltiplas demandas ordenadas por dependência técnica
    (mecanismo legado, desativado desde o Ato 1/2 — ver nota em
    ORDEM_RESOLUCAO acima).
    """
    sessao_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    modo_execucao: ModoExecucao = ModoExecucao.GPU_LOCAL
    camada_rag: CamadaRAG = CamadaRAG.PUBLICA

    demandas: List[DemandState] = Field(default_factory=list)
    indice_ativo: int = 0   # qual demanda está sendo trabalhada agora

    # Última pergunta formulada pelo agente — usada no próximo turno para
    # contextualizar a extração de campos (o Qwen sabe a qual campo a resposta se refere)
    ultima_pergunta_agente: str = ""

    created_at: datetime = Field(default_factory=datetime.now)

    @property
    def demanda_ativa(self) -> Optional[DemandState]:
        """Retorna a demanda sendo trabalhada no momento."""
        if not self.demandas:
            return None
        return self.demandas[self.indice_ativo]

    def adicionar_demanda(self, demanda: DemandState) -> None:
        """
        Adiciona uma demanda e reordena pela cadeia de dependência
        (mecanismo legado, desativado desde o Ato 1/2 — ver nota em
        ORDEM_RESOLUCAO acima).
        """
        self.demandas.append(demanda)
        self.demandas = ordenar_demandas(self.demandas)
        # Reposiciona o índice ativo para a primeira demanda não concluída
        for i, d in enumerate(self.demandas):
            if d.readiness != ReadinessStatus.PRONTA:
                self.indice_ativo = i
                break

    def avancar_demanda(self) -> bool:
        """
        Avança para a próxima demanda da fila.
        Retorna True se há próxima, False se todas foram concluídas.
        """
        if self.indice_ativo < len(self.demandas) - 1:
            self.indice_ativo += 1
            return True
        return False

    def todas_concluidas(self) -> bool:
        return all(
            d.readiness == ReadinessStatus.PRONTA
            for d in self.demandas
        )


# ────────────────────────────────────────────────────────────
# BRIEFING OUTPUT
# Saída final gerada após aprovação humana.
# ────────────────────────────────────────────────────────────

class BriefingOutput(BaseModel):
    """
    Briefing estruturado gerado após aprovação explícita do usuário.
    Nunca enviado automaticamente — sempre download manual.
    Espelha exatamente os campos da lista SharePoint da YDUQS.
    """
    # Campos do sistema SharePoint
    titulo:                    str
    status:                    StatusDemanda
    tipo_demanda:              TipoDemanda
    classificacao_estrategica: List[ClassificacaoEstrategica]
    valor_negocio:             ValorNegocio
    objetivo:                  str
    resultado_esperado:        str
    perguntas_de_negocio:      List[str]
    bloqueios:                 Optional[str] = None
    link_evidencia:            Optional[str] = None
    resumo_executivo:          Optional[str] = None   # texto usado pelo TTS (Piper)

    # Campos de rastreabilidade do DataBrief AI
    id_demanda:   str
    sessao_id:    str
    completude:   float                           # 0.0 a 1.0
    pendencias:   List[str] = Field(default_factory=list)
    gerado_em:    datetime = Field(default_factory=datetime.now)
    modo_execucao: ModoExecucao = ModoExecucao.GPU_LOCAL

    # Uso acumulado da API da OpenAI NESTA SESSÃO do Colab (chamadas feitas,
    # limite configurado, tokens totais) — só preenchido quando modo_execucao
    # == OPENAI; None nos modos locais, onde não existe chamada de API pra
    # contar. Vem de graph.agent.obter_uso_openai(); pedido do Phil (12/09)
    # depois de confirmar ao vivo que o contador só aparecia no console —
    # rede de segurança/transparência de custo, já que é uma chave
    # institucional sem painel de billing próprio.
    uso_openai: Optional[Dict[str, Any]] = None

    # Proveniência completa de cada campo (para o painel)
    proveniencia: Dict[str, FieldProvenance] = Field(default_factory=dict)

    @classmethod
    def from_demand_state(
        cls, state: DemandState, modo: ModoExecucao,
        uso_openai: Optional[Dict[str, Any]] = None,
    ) -> "BriefingOutput":
        """
        Constrói o BriefingOutput a partir do DemandState aprovado.
        Chamado apenas após aprovação explícita do usuário.

        `uso_openai` é opcional e só deveria vir preenchido (dict, ver
        graph.agent.obter_uso_openai()) quando o modo ativo é OPENAI — quem
        chama decide isso, aqui só é repassado ao model_dump final.
        """
        def valor(fp: Optional[FieldProvenance]) -> Any:
            return fp.valor if fp else None

        prov: Dict[str, FieldProvenance] = {}
        for campo in ["titulo", "objetivo", "resultado_esperado", "bloqueios",
                      "link_evidencia"]:
            fp = getattr(state, campo, None)
            if fp:
                prov[campo] = fp
        # Bloco 30 — perguntas_de_negocio e os 3 campos de opção fixa também
        # entram na proveniência exportada (antes ficavam de fora do JSON).
        if state.perguntas_de_negocio:
            prov["perguntas_de_negocio"] = state.perguntas_de_negocio[0]
        for campo in ["tipo_demanda", "valor_negocio", "classificacao_estrategica"]:
            fp = state.proveniencia_campos_fixos.get(campo)
            if fp:
                prov[campo] = fp

        return cls(
            titulo=valor(state.titulo) or "",
            status=state.status,
            tipo_demanda=state.tipo_demanda,
            classificacao_estrategica=state.classificacao_estrategica,
            valor_negocio=state.valor_negocio,
            objetivo=valor(state.objetivo) or "",
            resultado_esperado=valor(state.resultado_esperado) or "",
            perguntas_de_negocio=[fp.valor for fp in state.perguntas_de_negocio],
            bloqueios=valor(state.bloqueios),
            link_evidencia=valor(state.link_evidencia),
            resumo_executivo=state.resumo_executivo,
            id_demanda=state.id_demanda,
            sessao_id=state.sessao_id,
            completude=state.calcular_completude(),
            pendencias=state.pendencias,
            modo_execucao=modo,
            uso_openai=uso_openai,
            proveniencia=prov,
        )

    def to_sharepoint_dict(self) -> Dict[str, Any]:
        """
        Exporta os campos no formato esperado pelo SharePoint da YDUQS.
        Usado para geração do JSON para download.
        """
        return {
            "Título":                   self.titulo,
            "Status":                   self.status.value,
            "Tipo_Demanda":             self.tipo_demanda.value,
            "Classificacao_Estrategica": [c.value for c in self.classificacao_estrategica],
            "Valor_Negocio":            self.valor_negocio.value if self.valor_negocio else None,
            "Objetivo":                 self.objetivo,
            "Resultado":                self.resultado_esperado,
            "Perguntas_de_Negocio":     self.perguntas_de_negocio,
            "Bloqueios":                self.bloqueios,
            "Link_Evidencia":           self.link_evidencia,
        }
