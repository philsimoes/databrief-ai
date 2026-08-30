# ============================================================
# testes/casos_teste.py
# DataBrief AI — Ato 3, Bloco 09: Casos de teste sintéticos
#
# Corpus sintético baseado em contexto REAL da YDUQS — holding de
# Instituições de Ensino Superior (marcas Estácio, Wyden, Ibmec,
# IDOMED, Damásio, Grupo Q/Qconcursos, Hardwork Medicina; modalidades
# Presencial, Semipresencial e Digital/EaD; +2.000 polos de EaD,
# +100 campi; segmentos de reporte Premium/EaD/Presencial).
# Substitui a decisão anterior de usar a empresa fictícia "Aurora
# Varejo" — pedido do Phil (30/08): os casos devem tratar de questões
# da YDUQS de verdade, não de um setor genérico.
#
# IMPORTANTE — o que muda e o que NÃO muda em relação à decisão de
# "totalmente sintéticos" (30/08, mantida): os cenários abaixo usam
# marcas, modalidades, métricas e estrutura REAIS da YDUQS (extraídos
# de claude/corpus_yduqs_*.md e do PDF de apresentação corporativa,
# já no projeto), mas as DEMANDAS em si continuam inventadas — não
# são baseadas em nenhum chamado, ticket ou pedido real que o Phil
# tenha recebido no trabalho. É o mesmo espírito de antes (mostrar
# bem as features do agente), só que com vocabulário e contexto de
# negócio verdadeiros em vez de uma empresa genérica.
#
# Cada caso é rodado programaticamente por scripts/avaliar.py via
# processar_turno(), turno a turno, sem interação manual no Gradio.
# O script compara o DemandState resultante com "campos_esperados"
# e a métrica de turnos-até-prontidão com "turnos_ate_pronta_esperado".
#
# Schema de cada caso:
#   id                          — ex: "C001"
#   categoria                   — tag usada para agrupar métricas por cenário
#   descricao                   — o que o caso testa, em uma frase
#   turnos                      — lista ordenada de entradas do usuário:
#       tipo       — "text" | "audio" | "file" (== TipoInput)
#       conteudo   — texto (ou transcrição simulada, ou texto extraído do anexo)
#       nome_arquivo — só quando tipo == "file" (propagado a FieldProvenance.arquivo)
#   campos_esperados_apos_turnos — valores que devem estar preenchidos
#       SOMENTE a partir dos turnos acima (antes de qualquer pergunta de
#       fallback do agente) — usa os .value dos enums, como aparecem no
#       briefing final
#   readiness_esperado_apos_turnos — ReadinessStatus esperado logo após
#       os turnos acima serem processados (antes de qualquer resposta a
#       perguntas de esclarecimento) — "pronta" só quando o caso já
#       fornece todos os campos obrigatórios nos próprios turnos
#   turnos_ate_pronta_esperado — int | None — só preenchido quando o
#       caso é fechado (roteiro completo até PRONTA, incluindo eventuais
#       respostas a perguntas fixas simuladas em turnos_fallback)
#   turnos_fallback — respostas simuladas a perguntas de esclarecimento
#       que o agente deve fazer (mesmo schema de "turnos"); usado só nos
#       casos fechados
#   observacoes — nota livre: bug/decisão de design que o caso documenta
# ============================================================

CASOS = [
    # ────────────────────────────────────────────────────────────
    # 1. Tema sem formato — só o assunto, nenhuma pista de tipo_demanda.
    #    Deve cair em esclarecimento e disparar o Radio de tipo_demanda
    #    (Bloco 08) em vez de depender do Qwen adivinhar em texto livre.
    # ────────────────────────────────────────────────────────────
    {
        "id": "C001",
        "categoria": "tema_sem_formato",
        "descricao": "Demanda menciona só o tema (evasão), sem indicar formato de entrega — deve pedir tipo_demanda via Radio",
        "turnos": [
            {"tipo": "text", "conteudo": "Preciso entender melhor a evasão de alunos no curso de Enfermagem da Estácio."},
        ],
        "campos_esperados_apos_turnos": {},
        "readiness_esperado_apos_turnos": "esclarecimento",
        "turnos_ate_pronta_esperado": None,
        "turnos_fallback": [],
        "observacoes": "objetivo pode ser inferido do texto, mas tipo_demanda fica None — este é o caso que motivou o Bloco 08 (Radio de tipo_demanda). Enfermagem citada de propósito: passou a ser exclusiva do presencial no novo marco regulatório, tema real e atual da YDUQS.",
    },

    # ────────────────────────────────────────────────────────────
    # 2. Produto de Dados explícito — "dashboard" mencionado direto.
    #    Caminho feliz de extração multi-campo em um turno só.
    # ────────────────────────────────────────────────────────────
    {
        "id": "C002",
        "categoria": "formato_explicito",
        "descricao": "Demanda pede um dashboard de forma explícita, com objetivo e público-alvo claros",
        "turnos": [
            {
                "tipo": "text",
                "conteudo": (
                    "Preciso de um dashboard para a diretoria comercial de Estácio & Wyden acompanhar "
                    "a captação do Semipresencial por polo de EaD, atualizado toda semana."
                ),
            },
        ],
        "campos_esperados_apos_turnos": {
            "tipo_demanda": "Produto de Dados",
            "objetivo": "acompanhar a captação do Semipresencial por polo de EaD",
        },
        "readiness_esperado_apos_turnos": "esclarecimento",
        "turnos_ate_pronta_esperado": None,
        "turnos_fallback": [],
        "observacoes": "tipo_demanda deve vir preenchido pelo Qwen (PROMPT_EXTRAIR mapeia 'dashboard' → Produto de Dados) sem precisar do Radio. Semipresencial é a modalidade que mais cresce na YDUQS (CAGR de 41% segundo a apresentação corporativa) — tema de peso real.",
    },

    # ────────────────────────────────────────────────────────────
    # 3. Estruturante com bloqueio e link de evidência mencionados
    #    no mesmo turno — testa extração de campos opcionais.
    # ────────────────────────────────────────────────────────────
    {
        "id": "C003",
        "categoria": "bloqueio_e_link_mencionados",
        "descricao": "Demanda Estruturante já cita um bloqueio (dependência de outra área) e um link de evidência",
        "turnos": [
            {
                "tipo": "text",
                "conteudo": (
                    "Precisamos estruturar a camada Gold de matrículas a partir da Silver de captação, "
                    "com atualização diária, pra viabilizar os dashboards de Estácio & Wyden. Está "
                    "bloqueado porque depende da liberação de acesso da equipe de Engenharia — abri o "
                    "chamado aqui: https://chamados.yduqs.com.br/TICKET-4521"
                ),
            },
        ],
        "campos_esperados_apos_turnos": {
            "tipo_demanda": "Estruturante",
            "bloqueios": "depende da liberação de acesso da equipe de Engenharia",
            "link_evidencia": "https://chamados.yduqs.com.br/TICKET-4521",
        },
        "readiness_esperado_apos_turnos": "esclarecimento",
        "turnos_ate_pronta_esperado": None,
        "turnos_fallback": [],
        "observacoes": "bloqueios e link_evidencia são opcionais no schema — este caso confirma que, quando mencionados, entram com origem TEXT e não ficam de fora do briefing.",
    },

    # ────────────────────────────────────────────────────────────
    # 4. Alarmística completa em um único turno rico — testa quantos
    #    campos o Qwen consegue extrair de uma vez sem perguntas.
    # ────────────────────────────────────────────────────────────
    {
        "id": "C004",
        "categoria": "extracao_rica_um_turno",
        "descricao": "Demanda Alarmística com quase todos os campos universais descritos em um único turno",
        "turnos": [
            {
                "tipo": "text",
                "conteudo": (
                    "Título: Alerta de evasão no Digital. Preciso de um alerta automático quando a taxa "
                    "de evasão no ensino Digital passar de 8% em qualquer marca, verificado a cada 6 "
                    "horas, pra diretoria de operações agir antes de fechar o trimestre. É uma demanda "
                    "tática, ligada a eficiência operacional."
                ),
            },
        ],
        "campos_esperados_apos_turnos": {
            "titulo": "Alerta de evasão no Digital",
            "tipo_demanda": "Alarmística",
            "valor_negocio": "Tático",
            "classificacao_estrategica": ["Eficiência Operacional"],
        },
        "readiness_esperado_apos_turnos": "esclarecimento",
        "turnos_ate_pronta_esperado": None,
        "turnos_fallback": [],
        "observacoes": "resultado_esperado e perguntas_de_negocio ainda ficam pendentes — mede quantos dos 7 campos universais o Qwen extrai de um turno denso (métrica de recall por campo).",
    },

    # ────────────────────────────────────────────────────────────
    # 5. Análise com objetivo preenchido — regra especial de
    #    inferência: resultado_esperado NÃO deve ser perguntado de
    #    novo (campos_vazios linha 343-350 do agent.py).
    # ────────────────────────────────────────────────────────────
    {
        "id": "C005",
        "categoria": "inferencia_analise_resultado_esperado",
        "descricao": "Demanda de Análise com objetivo claro — resultado_esperado deve ser inferido automaticamente, sem pergunta extra",
        "turnos": [
            {
                "tipo": "text",
                "conteudo": (
                    "Quero uma análise pontual comparando a taxa de conversão do funil de captação do "
                    "Semipresencial vs. Presencial em Estácio & Wyden nos últimos 6 meses."
                ),
            },
        ],
        "campos_esperados_apos_turnos": {
            "tipo_demanda": "Análise",
            "objetivo": "comparando a taxa de conversão do funil de captação do Semipresencial vs. Presencial",
        },
        "readiness_esperado_apos_turnos": "esclarecimento",
        "turnos_ate_pronta_esperado": None,
        "turnos_fallback": [],
        "observacoes": "campo crítico deste caso: resultado_esperado NÃO deve aparecer em campos_vazios() após este turno, mesmo sem ter sido dito explicitamente — é a regra de inferência documentada no docstring de campos_vazios().",
    },

    # ────────────────────────────────────────────────────────────
    # 6. Roteiro fechado, campos opcionais nunca mencionados —
    #    briefing deve fechar em PRONTA sem bloqueios/link_evidencia.
    # ────────────────────────────────────────────────────────────
    {
        "id": "C006",
        "categoria": "fechado_sem_campos_opcionais",
        "descricao": "Roteiro completo até PRONTA sem nunca mencionar bloqueios ou link de evidência",
        "turnos": [
            {"tipo": "text", "conteudo": "Preciso de um relatório automatizado mensal com a taxa de renovação de matrícula da pós-graduação do Ibmec, pro time de retenção."},
        ],
        "campos_esperados_apos_turnos": {
            "tipo_demanda": "Produto de Dados",
        },
        "readiness_esperado_apos_turnos": "esclarecimento",
        "turnos_ate_pronta_esperado": 4,
        "turnos_fallback": [
            {"tipo": "text", "conteudo": "Reduzir a queda de renovação identificando os padrões de não-renovação com 2 meses de antecedência."},
            {"tipo": "text", "conteudo": "Tático, ligado a monitoramento."},
            {"tipo": "text", "conteudo": "Quais motivos de não-renovação mais pesam? Existe correlação com o polo de origem do aluno?"},
        ],
        "observacoes": "bloqueios e link_evidencia devem ficar None no briefing final e NÃO devem contar como pendência — confirma que campos opcionais não bloqueiam readiness == PRONTA. Ibmec é segmento Premium, com 95% de taxa de renovação segundo a apresentação corporativa — bom cenário pra mostrar monitoramento de um indicador já forte.",
    },

    # ────────────────────────────────────────────────────────────
    # 7. Anexo (PDF/DOCX/TXT) — testa origem ATTACHMENT e propagação
    #    de nome_arquivo para FieldProvenance.arquivo.
    # ────────────────────────────────────────────────────────────
    {
        "id": "C007",
        "categoria": "anexo_longo",
        "descricao": "Demanda enviada em texto curto + um anexo longo que traz o objetivo detalhado",
        "turnos": [
            {"tipo": "text", "conteudo": "Segue o briefing que recebi do time comercial do Ibmec, anexei o documento."},
            {
                "tipo": "file",
                "nome_arquivo": "briefing_comercial_ibmec.docx",
                "conteudo": (
                    "Contexto: a diretoria comercial do Ibmec quer entender por que a taxa de renovação "
                    "de matrícula da pós-graduação caiu no último trimestre em São Paulo, praça que "
                    "lidera a receita da graduação. Objetivo: mapear os segmentos de aluno com maior "
                    "queda de renovação e propor gatilhos de retenção. Este é um pedido de análise "
                    "pontual, não recorrente."
                ),
            },
        ],
        "campos_esperados_apos_turnos": {
            "tipo_demanda": "Análise",
            "objetivo": "mapear os segmentos de aluno com maior queda de renovação e propor gatilhos de retenção",
        },
        "readiness_esperado_apos_turnos": "esclarecimento",
        "turnos_ate_pronta_esperado": None,
        "turnos_fallback": [],
        "observacoes": "campo extraído do anexo deve registrar origem=ATTACHMENT e arquivo='briefing_comercial_ibmec.docx' no FieldProvenance — confirma no painel de proveniência.",
    },

    # ────────────────────────────────────────────────────────────
    # 8. Turno de áudio — transcrição simulada (Whisper já rodou),
    #    testa origem AUDIO e timestamp_audio.
    # ────────────────────────────────────────────────────────────
    {
        "id": "C008",
        "categoria": "turno_audio",
        "descricao": "Demanda enviada por áudio (transcrição simulada, como se o Whisper já tivesse rodado)",
        "turnos": [
            {
                "tipo": "audio",
                "conteudo": (
                    "Oi, aqui é o Phil. Eu preciso de um painel de monitoramento pra acompanhar em "
                    "tempo real a fila da Central de Relacionamento do aluno da Estácio, porque a "
                    "diretoria quer saber se o tempo médio de espera tá dentro da meta."
                ),
            },
        ],
        "campos_esperados_apos_turnos": {
            "tipo_demanda": "Produto de Dados",
            "objetivo": "acompanhar em tempo real a fila da Central de Relacionamento do aluno",
        },
        "readiness_esperado_apos_turnos": "esclarecimento",
        "turnos_ate_pronta_esperado": None,
        "turnos_fallback": [],
        "observacoes": "no ambiente real este conteúdo viria de transcrever_audio(); aqui já entra como texto transcrito porque o script de avaliação não invoca o Whisper — o que se testa é o pipeline a partir da transcrição (origem=AUDIO, timestamp_audio setado).",
    },

    # ────────────────────────────────────────────────────────────
    # 9. Múltiplas perguntas de negócio na mesma demanda — testa
    #    extração de lista em perguntas_de_negocio (List[FieldProvenance]).
    # ────────────────────────────────────────────────────────────
    {
        "id": "C009",
        "categoria": "multiplas_perguntas_negocio",
        "descricao": "Usuário já traz várias perguntas de negócio no mesmo turno em vez de uma só",
        "turnos": [
            {
                "tipo": "text",
                "conteudo": (
                    "Quero uma análise da campanha de captação do vestibular de Medicina do IDOMED. As "
                    "perguntas que a diretoria quer responder são: qual escola médica teve maior taxa "
                    "de preenchimento de vagas, qual canal trouxe mais conversão, e se o desconto médio "
                    "aplicado nas vagas adicionais valeu a pena olhando o ticket médio."
                ),
            },
        ],
        "campos_esperados_apos_turnos": {
            "tipo_demanda": "Análise",
        },
        "readiness_esperado_apos_turnos": "esclarecimento",
        "turnos_ate_pronta_esperado": None,
        "turnos_fallback": [],
        "observacoes": "espera-se 3 entradas em perguntas_de_negocio (uma por pergunta identificada), não uma string única concatenada — mede a qualidade de segmentação do Qwen nesta lista. Vagas adicionais de Medicina são um tema real de crescimento do IDOMED (alta taxa de aprovação nessas vagas, segundo a apresentação corporativa).",
    },

    # ────────────────────────────────────────────────────────────
    # 10. Demanda vaga total — discovery profundo, sem tema nem
    #     formato, só um pedido genérico de ajuda com dados.
    # ────────────────────────────────────────────────────────────
    {
        "id": "C010",
        "categoria": "demanda_vaga_total",
        "descricao": "Pedido inicial sem tema, sem formato, sem contexto — deve cair em discovery e não travar",
        "turnos": [
            {"tipo": "text", "conteudo": "Oi, preciso de uma ajuda com uns dados aqui na YDUQS."},
        ],
        "campos_esperados_apos_turnos": {},
        "readiness_esperado_apos_turnos": "discovery",
        "turnos_ate_pronta_esperado": None,
        "turnos_fallback": [],
        "observacoes": "nenhum campo obrigatório é extraível deste turno — confirma que o agente não força PRONTA nem trava, e que a pergunta seguinte é de descoberta (não uma das PERGUNTAS_FIXAS de campo específico).",
    },
]


def resumo_categorias() -> dict:
    """Conta quantos casos existem por categoria — usado por scripts/avaliar.py
    para checar cobertura antes de rodar a suíte completa."""
    contagem: dict = {}
    for caso in CASOS:
        contagem[caso["categoria"]] = contagem.get(caso["categoria"], 0) + 1
    return contagem


if __name__ == "__main__":
    for categoria, qtd in resumo_categorias().items():
        print(f"{categoria}: {qtd} caso(s)")
    print(f"\nTotal: {len(CASOS)} casos (lote 1 de aprovação — meta final: 50)")
