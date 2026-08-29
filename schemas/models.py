# ============================================================
# schemas/models.py
# DataBrief AI — Bloco 01: Schemas Pydantic
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
    """Tipo técnico da entrega."""
    ANALISE        = "Análise"
    ESTRUTURANTE   = "Estruturante"
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
    BLOQUEADA     = "bloqueada"     # dependência externa impede avanço


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
# Produto de Dados e Alarmística sempre dependem de Estruturante.
# Análise depende de Estruturante apenas se usar Gold como fonte.
# A dependência condicional da Análise é resolvida no grafo,
# não aqui — aqui só definimos a ordem padrão.
# ────────────────────────────────────────────────────────────

ORDEM_RESOLUCAO: Dict[TipoDemanda, int] = {
    TipoDemanda.ESTRUTURANTE:  0,
    TipoDemanda.ANALISE:       1,
    TipoDemanda.PRODUTO_DADOS: 2,
    TipoDemanda.ALARMASTICA:   2,
}

DEPENDE_SEMPRE_DE_ESTRUTURANTE = {
    TipoDemanda.PRODUTO_DADOS,
    TipoDemanda.ALARMASTICA,
}

FONTES_GOLD = {"gold", "Gold", "GOLD"}


def analise_depende_de_gold(demanda: "DemandState") -> bool:
    if demanda.campos_analise and demanda.campos_analise.fonte_dados:
        fonte = str(demanda.campos_analise.fonte_dados.valor)
        return any(g in fonte for g in FONTES_GOLD)
    return False


def ordenar_demandas(demandas: List["DemandState"]) -> List["DemandState"]:
    def chave(d: "DemandState") -> int:
        if d.tipo_demanda == TipoDemanda.ANALISE and analise_depende_de_gold(d):
            return 2
        return ORDEM_RESOLUCAO.get(d.tipo_demanda, 99)
    return sorted(demandas, key=chave)

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
# CAMPOS ESPECÍFICOS POR TIPO DE DEMANDA
# ────────────────────────────────────────────────────────────

class CamposAnalise(BaseModel):
    """Campos obrigatórios para demandas do tipo Análise."""
    recorte_temporal: Optional[FieldProvenance] = None   # ex: "últimos 6 meses"
    estratificacoes: Optional[FieldProvenance] = None    # ex: "por curso, marca, polo"
    fonte_dados: Optional[FieldProvenance] = None        # ex: "Silver matrícula"
    metrica_principal: Optional[FieldProvenance] = None  # ex: "taxa de evasão"


class CamposEstruturante(BaseModel):
    """Campos obrigatórios para demandas do tipo Estruturante."""
    camada_origem: Optional[FieldProvenance] = None      # ex: "Silver"
    camada_destino: Optional[FieldProvenance] = None     # ex: "Gold"
    fonte_dados: Optional[FieldProvenance] = None
    frequencia_atualizacao: Optional[FieldProvenance] = None  # ex: "diária"


class CamposProdutoDados(BaseModel):
    """Campos obrigatórios para demandas do tipo Produto de Dados."""
    ferramenta: Optional[FieldProvenance] = None         # ex: "PowerBI", "Databricks AIBI", "Agente"
    publico_alvo: Optional[FieldProvenance] = None       # ex: "diretoria comercial"
    fonte_dados: Optional[FieldProvenance] = None
    frequencia_atualizacao: Optional[FieldProvenance] = None


class CamposAlarmastica(BaseModel):
    """Campos obrigatórios para demandas do tipo Alarmística."""
    indicador_monitorado: Optional[FieldProvenance] = None   # ex: "taxa de sucesso de captura"
    threshold_alerta: Optional[FieldProvenance] = None       # ex: "abaixo de 85%"
    frequencia_verificacao: Optional[FieldProvenance] = None # ex: "a cada 6 horas"
    destinatarios_email: Optional[FieldProvenance] = None    # ex: ["phil@yduqs.com.br"]


# ────────────────────────────────────────────────────────────
# DEMAND STATE
# Estado completo de uma única demanda ao longo da conversa.
# ────────────────────────────────────────────────────────────

class DemandState(BaseModel):
    """
    Estado completo de uma demanda.
    Todos os campos rastreiam proveniência individual.
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

    # Campos específicos por tipo (apenas um será populado)
    campos_analise:       Optional[CamposAnalise]      = None
    campos_estruturante:  Optional[CamposEstruturante] = None
    campos_produto_dados: Optional[CamposProdutoDados] = None
    campos_alarmastica:   Optional[CamposAlarmastica]  = None

    # Estado do agente
    readiness:      ReadinessStatus = ReadinessStatus.DISCOVERY
    pendencias:     List[str]       = Field(default_factory=list)
    turno_atual:    int             = 0
    historico_turnos: List[TurnInput] = Field(default_factory=list)

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

    def campos_obrigatorios_preenchidos(self) -> bool:
        """
        Verifica se os campos universais obrigatórios estão presentes.
        Campos específicos por tipo são verificados separadamente.
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
    Pode conter múltiplas demandas ordenadas por dependência técnica.
    Ex: Gold (Estruturante) → Agente (Produto de Dados).
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
        Adiciona uma demanda e reordena pela cadeia de dependência.
        Estruturante sempre fica antes de Produto de Dados e Alarmística.
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

    # Proveniência completa de cada campo (para o painel)
    proveniencia: Dict[str, FieldProvenance] = Field(default_factory=dict)

    @classmethod
    def from_demand_state(cls, state: DemandState, modo: ModoExecucao) -> "BriefingOutput":
        """
        Constrói o BriefingOutput a partir do DemandState aprovado.
        Chamado apenas após aprovação explícita do usuário.
        """
        def valor(fp: Optional[FieldProvenance]) -> Any:
            return fp.valor if fp else None

        prov: Dict[str, FieldProvenance] = {}
        for campo in ["titulo", "objetivo", "resultado_esperado", "bloqueios",
                      "link_evidencia"]:
            fp = getattr(state, campo, None)
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
