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
# Cada caso é rodado programaticamente via processar_turno(), turno a
# turno, sem interação manual no Gradio. Isso é feito por
# scripts/rodar_casos.py (runner, Bloco 10/13) — que agora (Bloco 25)
# consegue fechar TODO caso até PRONTA, não só os antigos "casos
# fechados" — ver "respostas_por_campo agora universal" abaixo.
#
# Bloco 25 (07/09) — dois campos novos, presentes em TODOS os 10 casos:
#   respostas_por_campo — deixou de ser exclusivo de "casos fechados"
#       (a minoria, como o antigo C006). Bloco 24 criou
#       processar_confirmacao_tipo_demanda()/processar_confirmacao_
#       resultado_esperado() em graph/agent.py (mesmo padrão que já
#       existia pra valor_negocio) — o runner agora consegue simular o
#       clique de QUALQUER Radio/checkbox sem depender do Gradio, então
#       todo caso ganhou respostas pra fechar até o fim, viabilizando
#       gabarito_final (abaixo) pra F1/recall no script de avaliação
#       (scripts/avaliar.py, Ato 3 item 3). Continua servindo pro papel
#       original também (achar bugs rodando ao vivo).
#       Valores por campo, conforme o mecanismo real de cada um:
#         tipo_demanda / resultado_esperado — valor exato do enum (o
#           runner chama processar_confirmacao_* direto, como um clique
#           real de Radio — nunca texto livre pro Qwen re-extrair,
#           mesmo motivo que tirou esses 2 campos da extração de texto
#           livre desde o fechamento do Ato 2/3).
#         classificacao_estrategica — lista de valores exatos do enum
#           (runner usa processar_selecao_checkbox direto, por
#           determinismo — embora esse campo ainda seja extraível em
#           texto livre também, diferente dos 2 acima).
#         valor_negocio — igual, valor exato do enum (já existia).
#         demais campos (objetivo, titulo) — texto livre normal.
#         perguntas_de_negocio NUNCA entra aqui — o runner trata esse
#           campo automaticamente (aceita a sugestão do Qwen), porque
#           esse campo nunca é preenchido por texto livre digitado.
#       Campos já esperados vir do(s) turno(s) principal(is) (ver
#       campos_esperados_apos_turnos) ganham uma resposta aqui mesmo
#       assim, como rede de segurança — se a extração não pegar numa
#       rodada ao vivo (o Qwen é não-determinístico), o caso ainda
#       fecha em vez de travar no limite de segurança.
#   gabarito_final — valor esperado de cada um dos 7 campos universais
#       no ESTADO FINAL (depois de fechado até PRONTA) — usado pelo F1
#       por campo e pelo recall de lacunas (scripts/avaliar.py).
#       Campos categóricos (tipo_demanda, valor_negocio,
#       classificacao_estrategica, e resultado_esperado quando
#       tipo_demanda == Produto de Dados) são comparados por igualdade
#       exata — F1 automático de verdade. resultado_esperado quando
#       tipo_demanda == Análise é "Análise: {objetivo}" — só a
#       ESTRUTURA (prefixo + origem RULE) é checada automaticamente, o
#       texto do objetivo embutido varia e entra na revisão humana.
#       Campos de texto livre (titulo, objetivo, bloqueios,
#       link_evidencia) são reportados lado a lado (gabarito vs.
#       obtido) mas a correção é julgada por você no mesmo mecanismo de
#       veredito da concordância humana — comparar string exata com um
#       LLM não-determinístico seria enganoso. perguntas_de_negocio é
#       só informacional (não entra no F1).
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
#       briefing final.
#       IMPORTANTE (achado do 1º teste ao vivo, 30/08, caso C004): NUNCA
#       coloque "titulo" aqui. Igual a perguntas_de_negocio, titulo tem uma
#       guarda anti-alucinação em aplicar_extracao() (graph/agent.py) que só
#       aceita o valor quando a ÚLTIMA PERGUNTA DO AGENTE continha
#       "chamaria"/"nome"/"título"/"titulo" — ou seja, só depois que
#       PERGUNTAS_FIXAS["titulo"] já foi feita. Nos turnos principais (antes
#       de qualquer pergunta do agente) essa condição nunca é satisfeita,
#       então titulo SEMPRE vem None nesse ponto — não importa o quão
#       explícito o texto do usuário seja (ex: "Título: X." logo no início).
#       Só cabe em campos_esperados_apos_turnos depois de um turno de
#       resposta que veio logo após essa pergunta específica (ou, nos casos
#       fechados, dentro de respostas_por_campo — ver C006).
#   readiness_esperado_apos_turnos — ReadinessStatus esperado logo após
#       os turnos acima serem processados (antes de qualquer resposta a
#       perguntas de esclarecimento) — "pronta" só quando o caso já
#       fornece todos os campos obrigatórios nos próprios turnos.
#       IMPORTANTE (achado do 1º teste ao vivo, 30/08): no_avaliar_completude()
#       em graph/agent.py só atribui ReadinessStatus.PRONTA ou .DISCOVERY —
#       .ESCLARECIMENTO existe no enum (schemas/models.py) mas NUNCA é
#       atribuído em lugar nenhum do grafo hoje. Todo caso aberto (que não
#       chega em PRONTA) sempre reporta "discovery", nunca "esclarecimento" —
#       por isso é o único valor usado abaixo além de "pronta".
#   respostas_por_campo — dict campo→texto, só presente nos casos "fechados"
#       (a minoria, pensados pra chegar em PRONTA). O runner (scripts/
#       rodar_casos.py) lê o campo_prioritario_atual que o agente devolve a
#       cada turno e busca a resposta certa nesse dict — não é uma lista de
#       turnos na ordem fixa, porque a ordem real depende de quanto o Qwen
#       já extraiu do turno principal (não dá pra prever com certeza sem
#       rodar ao vivo). O campo "perguntas_de_negocio" é tratado à parte
#       pelo runner (nunca por texto livre — ver observação em C006) e não
#       entra nesse dict.
#   turnos_ate_pronta_esperado — int | None — só nos casos fechados; é uma
#       ESTIMATIVA pro runner reportar lado a lado com o real, não uma
#       asserção rígida (Qwen é não-determinístico turno a turno).
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
        "readiness_esperado_apos_turnos": "discovery",
        "turnos_ate_pronta_esperado": 6,
        "respostas_por_campo": {
            "objetivo": "Entender os principais motivos da evasão de alunos no curso de Enfermagem da Estácio.",
            "tipo_demanda": "Análise",
            "valor_negocio": "Tático",
            "classificacao_estrategica": ["Insight para Decisão"],
            "titulo": "Evasão no curso de Enfermagem da Estácio",
        },
        "gabarito_final": {
            "titulo": "Evasão no curso de Enfermagem da Estácio",
            "tipo_demanda": "Análise",
            "objetivo": "Entender os principais motivos da evasão de alunos no curso de Enfermagem da Estácio.",
            "resultado_esperado": "Análise: Entender os principais motivos da evasão de alunos no curso de Enfermagem da Estácio.",
            "valor_negocio": "Tático",
            "classificacao_estrategica": ["Insight para Decisão"],
            "perguntas_de_negocio": ["Quais são os principais motivos da evasão no curso de Enfermagem da Estácio?"],
        },
        "observacoes": "objetivo pode ser inferido do texto, mas tipo_demanda fica None — este é o caso que motivou o Bloco 08 (Radio de tipo_demanda). Enfermagem citada de propósito: passou a ser exclusiva do presencial no novo marco regulatório, tema real e atual da YDUQS. Bloco 25: tipo_demanda escolhido como Análise pra fechamento (decisão de design do caso, não extração — o texto por si só não define isso, por design).",
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
        "readiness_esperado_apos_turnos": "discovery",
        "turnos_ate_pronta_esperado": 4,
        "respostas_por_campo": {
            # fallbacks — "dashboard" já é gatilho de PALAVRAS_FORMATO_EXPLICITO,
            # tipo_demanda e resultado_esperado devem vir de TEXT já no turno 1;
            # respostas aqui só entram em jogo se a extração falhar na rodada
            "tipo_demanda": "Produto de Dados",
            "resultado_esperado": "Dashboard interativo",
            "valor_negocio": "Estratégico",
            "classificacao_estrategica": ["Monitoramento"],
            "titulo": "Captação do Semipresencial por polo de EaD",
        },
        "gabarito_final": {
            "titulo": "Captação do Semipresencial por polo de EaD",
            "tipo_demanda": "Produto de Dados",
            "objetivo": "acompanhar a captação do Semipresencial por polo de EaD",
            "resultado_esperado": "Dashboard interativo",
            "valor_negocio": "Estratégico",
            "classificacao_estrategica": ["Monitoramento"],
            "perguntas_de_negocio": ["Qual polo de EaD está com a menor captação do Semipresencial?"],
        },
        "observacoes": "tipo_demanda deve vir preenchido pelo Qwen (PROMPT_EXTRAIR mapeia 'dashboard' → Produto de Dados) sem precisar do Radio. Semipresencial é a modalidade que mais cresce na YDUQS (CAGR de 41% segundo a apresentação corporativa) — tema de peso real. Bloco 25: valor_negocio escolhido como Estratégico (pedido é da diretoria comercial) — decisão de design do caso.",
    },

    # ────────────────────────────────────────────────────────────
    # 3. Produto de Dados (pipeline/tabela Gold como base de um dashboard)
    #    com bloqueio e link de evidência mencionados no mesmo turno —
    #    testa extração de campos opcionais.
    #
    #    Bloco 23 (06/09): esse caso testava a categoria "Estruturante",
    #    removida do TipoDemanda (decisão do Phil, como gerente de dados —
    #    pipeline/tabela Gold sempre serviam de base pra um produto/alerta
    #    final, nunca eram pedido de negócio isolado). Texto do turno não
    #    mudou — continua citando tanto "estruturar a camada Gold" quanto
    #    "dashboards de Estácio & Wyden" — só a expectativa de tipo_demanda
    #    mudou, porque as duas coisas agora caem na mesma categoria.
    # ────────────────────────────────────────────────────────────
    {
        "id": "C003",
        "categoria": "bloqueio_e_link_mencionados",
        "descricao": "Demanda Produto de Dados (pipeline/tabela Gold citados como base de um dashboard) já cita um bloqueio (dependência de outra área) e um link de evidência",
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
            "tipo_demanda": "Produto de Dados",
            "bloqueios": "depende da liberação de acesso da equipe de Engenharia",
            "link_evidencia": "https://chamados.yduqs.com.br/TICKET-4521",
        },
        "readiness_esperado_apos_turnos": "discovery",
        "turnos_ate_pronta_esperado": 5,
        "respostas_por_campo": {
            "objetivo": "estruturar a camada Gold de matrículas a partir da Silver de captação para viabilizar os dashboards de Estácio & Wyden",
            "tipo_demanda": "Produto de Dados",
            "resultado_esperado": "Dashboard interativo",
            "valor_negocio": "Tático",
            "classificacao_estrategica": ["Priorização"],
            "titulo": "Estruturação da camada Gold de matrículas Estácio & Wyden",
        },
        "gabarito_final": {
            "titulo": "Estruturação da camada Gold de matrículas Estácio & Wyden",
            "tipo_demanda": "Produto de Dados",
            "objetivo": "estruturar a camada Gold de matrículas a partir da Silver de captação para viabilizar os dashboards de Estácio & Wyden",
            "resultado_esperado": "Dashboard interativo",
            "valor_negocio": "Tático",
            "classificacao_estrategica": ["Priorização"],
            "bloqueios": "depende da liberação de acesso da equipe de Engenharia",
            "link_evidencia": "https://chamados.yduqs.com.br/TICKET-4521",
            "perguntas_de_negocio": ["Quando a equipe de Engenharia libera o acesso necessário?"],
        },
        "observacoes": "bloqueios e link_evidencia são opcionais no schema — este caso confirma que, quando mencionados, entram com origem TEXT e não ficam de fora do briefing. Nota (Bloco 23): expectativa de tipo_demanda mudou de 'Estruturante' pra 'Produto de Dados' — ver comentário do bloco acima.",
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
            "tipo_demanda": "Alarmística",
            "classificacao_estrategica": ["Eficiência Operacional"],
        },
        "readiness_esperado_apos_turnos": "discovery",
        "turnos_ate_pronta_esperado": 4,
        "respostas_por_campo": {
            "objetivo": "monitorar a taxa de evasão no ensino Digital para agir antes de fechar o trimestre",
            "tipo_demanda": "Alarmística",
            "valor_negocio": "Tático",
            "classificacao_estrategica": ["Eficiência Operacional"],
            "titulo": "Alerta de evasão no Digital",
        },
        "gabarito_final": {
            "titulo": "Alerta de evasão no Digital",
            "tipo_demanda": "Alarmística",
            "objetivo": "monitorar a taxa de evasão no ensino Digital para agir antes de fechar o trimestre",
            "resultado_esperado": "Alerta automático",
            "valor_negocio": "Tático",
            "classificacao_estrategica": ["Eficiência Operacional"],
            "perguntas_de_negocio": ["A taxa de evasão no Digital já ultrapassou 8% em alguma marca recentemente?"],
        },
        "observacoes": (
            "resultado_esperado e perguntas_de_negocio ainda ficam pendentes — mede quantos "
            "dos 7 campos universais o Qwen extrai de um turno denso (métrica de recall por "
            "campo). titulo NÃO entra em campos_esperados_apos_turnos mesmo vindo explícito "
            "logo no início do texto ('Título: Alerta de evasão no Digital.') — achado do 1º "
            "teste ao vivo (30/08): a guarda anti-alucinação de titulo em aplicar_extracao() só "
            "aceita o valor depois que o agente já perguntou especificamente sobre título (ver "
            "nota no cabeçalho deste arquivo); titulo sempre vem None neste checkpoint, por "
            "design, não por falha do Qwen. ATUALIZADO (fechamento Ato 2/3): valor_negocio "
            "também saiu de campos_esperados_apos_turnos — era justamente este o caso que "
            "expôs o bug real (Qwen extraía 'Operacional' em vez de 'Tático' aqui, 3/3 vezes "
            "em teste real, nos dois tamanhos de modelo, confundindo com a palavra 'operacional' "
            "dentro de 'eficiência operacional'). O campo deixou de ser extraído em texto livre "
            "e agora só é preenchido via confirmação de Radio — neste checkpoint (antes de "
            "qualquer Radio confirmado) ele sempre vem None, por design. Este caso continua "
            "sendo o motivo documentado da mudança; só o resultado esperado aqui mudou de "
            "'Tático' (o valor que deveria ter sido extraído) para 'não preenchido ainda' "
            "(o comportamento correto agora, que exige confirmação ativa do usuário)."
        ),
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
        "readiness_esperado_apos_turnos": "discovery",
        "turnos_ate_pronta_esperado": 5,
        "respostas_por_campo": {
            # fallback — só entra em jogo se a extração do turno principal falhar
            "objetivo": "comparando a taxa de conversão do funil de captação do Semipresencial vs. Presencial em Estácio & Wyden nos últimos 6 meses",
            "tipo_demanda": "Análise",
            "valor_negocio": "Tático",
            "classificacao_estrategica": ["Insight para Decisão"],
            "titulo": "Conversão do funil de captação: Semipresencial vs. Presencial",
            # resultado_esperado NÃO entra aqui — é inferido automaticamente por
            # RULE assim que tipo_demanda==Análise e objetivo existem (a regra
            # que este caso testa), nunca perguntado via Radio.
        },
        "gabarito_final": {
            "titulo": "Conversão do funil de captação: Semipresencial vs. Presencial",
            "tipo_demanda": "Análise",
            "objetivo": "comparando a taxa de conversão do funil de captação do Semipresencial vs. Presencial em Estácio & Wyden nos últimos 6 meses",
            "resultado_esperado": "Análise: comparando a taxa de conversão do funil de captação do Semipresencial vs. Presencial em Estácio & Wyden nos últimos 6 meses",
            "valor_negocio": "Tático",
            "classificacao_estrategica": ["Insight para Decisão"],
            "perguntas_de_negocio": ["Qual modalidade, Semipresencial ou Presencial, tem a maior taxa de conversão no funil de captação?"],
        },
        "observacoes": "campo crítico deste caso: resultado_esperado NÃO deve aparecer em campos_vazios() após este turno, mesmo sem ter sido dito explicitamente — é a regra de inferência documentada no docstring de campos_vazios(). Bloco 25: gabarito_final confirma a estrutura esperada ('Análise: ' + objetivo, origem RULE) — a comparação automática do script de avaliação deve checar só o prefixo/origem, não o texto do objetivo embutido (ver nota no cabeçalho do arquivo).",
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
        "readiness_esperado_apos_turnos": "discovery",
        "turnos_ate_pronta_esperado": 6,
        "respostas_por_campo": {
            "objetivo": "O objetivo é entender por que os alunos da pós-graduação do Ibmec não estão renovando a matrícula.",
            # CORRIGIDO (Bloco 25) — o comentário antigo aqui dizia que "relatório
            # automatizado" (texto do turno principal) NÃO estava na lista de
            # gatilhos de PALAVRAS_FORMATO_EXPLICITO. Isso deixou de ser verdade —
            # o dict hoje TEM "relatório automatizado": "Agente automatizado" como
            # chave exata (agent.py). Mas o preenchimento automático não depende só
            # da palavra bater no texto: o bloco inteiro só roda quando
            # dados.get("resultado_esperado") já vem preenchido pelo PRÓPRIO Qwen
            # na extração do turno — a checagem de palavra-chave é uma VALIDAÇÃO
            # contra alucinação, não um gatilho isolado (ver aplicar_extracao() em
            # agent.py, condição "if dados.get('resultado_esperado') and not
            # demanda.resultado_esperado"). Como isso depende do Qwen ser
            # não-determinístico, resultado_esperado pode vir preenchido via TEXT
            # já no turno 1 nesta rodada, ou não — a verificar ao vivo (não dá pra
            # confirmar sem rodar o modelo de verdade). A resposta abaixo serve de
            # rede de segurança pros dois casos. Valor precisa ser o EXATO do enum
            # (Bloco 25: resultado_esperado agora é sempre confirmação de Radio
            # pra Produto de Dados, nunca texto livre — ver processar_confirmacao_
            # resultado_esperado em agent.py).
            "resultado_esperado": "Agente automatizado",
            # valor_negocio só é aplicado via confirmação de Radio
            # (processar_confirmacao_valor_negocio) — o runner simula o clique,
            # então o valor aqui precisa bater EXATAMENTE com um dos 3 valores do
            # enum ValorNegocio (sem pontuação no final).
            "valor_negocio": "Tático",
            "classificacao_estrategica": ["Monitoramento"],
            "titulo": "Renovação de matrícula na pós-graduação do Ibmec",
        },
        "gabarito_final": {
            "titulo": "Renovação de matrícula na pós-graduação do Ibmec",
            "tipo_demanda": "Produto de Dados",
            "objetivo": "O objetivo é entender por que os alunos da pós-graduação do Ibmec não estão renovando a matrícula.",
            "resultado_esperado": "Agente automatizado",
            "valor_negocio": "Tático",
            "classificacao_estrategica": ["Monitoramento"],
            "bloqueios": None,
            "link_evidencia": None,
            "perguntas_de_negocio": ["Por que os alunos da pós-graduação do Ibmec não estão renovando a matrícula?"],
        },
        "observacoes": (
            "bloqueios e link_evidencia devem ficar None no briefing final e NÃO devem contar como pendência — "
            "confirma que campos opcionais não bloqueiam readiness == PRONTA. Ibmec é segmento Premium, com 95% "
            "de taxa de renovação segundo a apresentação corporativa — bom cenário pra mostrar monitoramento de "
            "um indicador já forte. IMPORTANTE: perguntas_de_negocio NUNCA é preenchido por texto livre — "
            "aplicar_extracao() ignora esse campo de propósito (ver comentário no próprio agent.py); o único "
            "caminho é a sugestão gerada pelo Qwen a partir do objetivo (PROMPT_SUGERIR_PERGUNTA), confirmada via "
            "processar_confirmacao_pergunta_negocio(). O runner (rodar_casos.py) trata isso automaticamente "
            "quando campo_prioritario_atual == 'perguntas_de_negocio' — não entra em respostas_por_campo. "
            "turnos_ate_pronta_esperado=6 é uma estimativa (1 principal + objetivo + resultado_esperado (se não "
            "vier de TEXT) + valor_negocio + classificacao_estrategica + perguntas_de_negocio [automático] + "
            "titulo) — pode variar bastante de verdade conforme o Qwen agrupar campos na mesma resposta ou "
            "acertar resultado_esperado via TEXT logo no turno 1 (ver nota acima)."
        ),
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
        "readiness_esperado_apos_turnos": "discovery",
        "turnos_ate_pronta_esperado": 6,
        "respostas_por_campo": {
            # fallback — só entra em jogo se a extração do anexo falhar
            "objetivo": "mapear os segmentos de aluno com maior queda de renovação e propor gatilhos de retenção",
            "tipo_demanda": "Análise",
            "valor_negocio": "Estratégico",
            "classificacao_estrategica": ["Insight para Decisão"],
            "titulo": "Queda de renovação na pós-graduação do Ibmec em São Paulo",
        },
        "gabarito_final": {
            "titulo": "Queda de renovação na pós-graduação do Ibmec em São Paulo",
            "tipo_demanda": "Análise",
            "objetivo": "mapear os segmentos de aluno com maior queda de renovação e propor gatilhos de retenção",
            "resultado_esperado": "Análise: mapear os segmentos de aluno com maior queda de renovação e propor gatilhos de retenção",
            "valor_negocio": "Estratégico",
            "classificacao_estrategica": ["Insight para Decisão"],
            "perguntas_de_negocio": ["Quais segmentos de aluno tiveram a maior queda de renovação de matrícula na pós-graduação do Ibmec em São Paulo?"],
        },
        "observacoes": "campo extraído do anexo deve registrar origem=ATTACHMENT e arquivo='briefing_comercial_ibmec.docx' no FieldProvenance — confirma no painel de proveniência. Bloco 25: valor_negocio escolhido como Estratégico (pedido vem da diretoria comercial, praça de São Paulo lidera receita da graduação) — decisão de design do caso, não extração.",
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
            "objetivo": "acompanhar em tempo real a fila da Central de Relacionamento do aluno",
        },
        "readiness_esperado_apos_turnos": "discovery",
        "turnos_ate_pronta_esperado": 6,
        "respostas_por_campo": {
            # tipo_demanda é decisão de design pro fechamento — o texto é
            # ambíguo de propósito (painel + monitoramento, ver observação
            # abaixo), então cai no Radio; Alarmística porque o pedido real é
            # comparar contra uma meta, não só visualizar.
            "tipo_demanda": "Alarmística",
            "objetivo": "acompanhar em tempo real a fila da Central de Relacionamento do aluno",
            "valor_negocio": "Operacional",
            "classificacao_estrategica": ["Monitoramento"],
            "titulo": "Monitoramento da fila da Central de Relacionamento do aluno",
            # resultado_esperado NÃO entra aqui — Alarmística sempre infere
            # "Alerta automático" por regra, nunca pergunta via Radio.
        },
        "gabarito_final": {
            "titulo": "Monitoramento da fila da Central de Relacionamento do aluno",
            "tipo_demanda": "Alarmística",
            "objetivo": "acompanhar em tempo real a fila da Central de Relacionamento do aluno",
            "resultado_esperado": "Alerta automático",
            "valor_negocio": "Operacional",
            "classificacao_estrategica": ["Monitoramento"],
            "perguntas_de_negocio": ["O tempo médio de espera da fila da Central de Relacionamento já ultrapassou a meta definida pela diretoria?"],
        },
        "observacoes": (
            "no ambiente real este conteúdo viria de transcrever_audio(); aqui já entra como texto "
            "transcrito porque o script de avaliação não invoca o Whisper — o que se testa é o "
            "pipeline a partir da transcrição (origem=AUDIO, timestamp_audio setado). ATUALIZADO "
            "(Bloco 18, fechamento Ato 2/3): tipo_demanda saiu de campos_esperados_apos_turnos de "
            "propósito. Esse é justamente o caso que expôs o bug real — 'painel de monitoramento' "
            "bate palavra-chave de Produto de Dados ('painel') E de Alarmística ('monitoramento') ao "
            "mesmo tempo, e em teste real (GPU_LOCAL) o Qwen respondeu confiante e ERRADO "
            "('Alarmística'). Agora, quando o texto bate com duas ou mais categorias ao mesmo tempo, "
            "o campo fica None neste checkpoint por design (nenhuma categoria trava sozinha) e só é "
            "resolvido depois, via Radio (Bloco 08) — mesmo princípio já aplicado a valor_negocio no "
            "C004: o resultado esperado aqui mudou de 'o valor que o Qwen deveria ter extraído' para "
            "'não preenchido ainda, aguardando confirmação ativa'."
        ),
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
        "readiness_esperado_apos_turnos": "discovery",
        "turnos_ate_pronta_esperado": 6,
        "respostas_por_campo": {
            # fallback — só entra em jogo se a extração do turno principal falhar
            "objetivo": "avaliar a campanha de captação do vestibular de Medicina do IDOMED, comparando taxa de preenchimento de vagas por escola médica, canal de conversão e o retorno do desconto nas vagas adicionais",
            "tipo_demanda": "Análise",
            "valor_negocio": "Tático",
            "classificacao_estrategica": ["Insight para Decisão"],
            "titulo": "Campanha de captação do vestibular de Medicina do IDOMED",
        },
        "gabarito_final": {
            "titulo": "Campanha de captação do vestibular de Medicina do IDOMED",
            "tipo_demanda": "Análise",
            "objetivo": "avaliar a campanha de captação do vestibular de Medicina do IDOMED, comparando taxa de preenchimento de vagas por escola médica, canal de conversão e o retorno do desconto nas vagas adicionais",
            "resultado_esperado": "Análise: avaliar a campanha de captação do vestibular de Medicina do IDOMED, comparando taxa de preenchimento de vagas por escola médica, canal de conversão e o retorno do desconto nas vagas adicionais",
            "valor_negocio": "Tático",
            "classificacao_estrategica": ["Insight para Decisão"],
            # lista de referência (informacional, não entra no F1) — o ideal é
            # que a sugestão do Qwen capture bem o contexto das 3 perguntas já
            # mencionadas pelo usuário, mesmo gerando só 1 pergunta de fato.
            "perguntas_de_negocio": [
                "Qual escola médica teve a maior taxa de preenchimento de vagas no vestibular de Medicina do IDOMED?",
                "Qual canal de captação trouxe mais conversão?",
                "O desconto médio aplicado nas vagas adicionais valeu a pena considerando o ticket médio?",
            ],
        },
        "observacoes": (
            "CORRIGIDO (era um erro no lote original): perguntas_de_negocio NUNCA é preenchido por extração de "
            "texto livre — mesmo o usuário listando 3 perguntas aqui, esse campo continua vazio após este turno "
            "(aplicar_extracao ignora esse campo de propósito). O objetivo deste caso não é testar extração de "
            "lista — é confirmar que, mais adiante (fora deste 1º lote, quando o roteiro chegar em "
            "campo_prioritario == 'perguntas_de_negocio'), a sugestão gerada pelo Qwen a partir do objetivo "
            "capture bem esse contexto de 3 perguntas distintas já mencionadas — métrica de qualidade pra medir "
            "no script de avaliação (item 3), não neste caso isolado. Vagas adicionais de Medicina são um tema "
            "real de crescimento do IDOMED (alta taxa de aprovação nessas vagas, segundo a apresentação "
            "corporativa)."
        ),
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
        "turnos_ate_pronta_esperado": 7,
        "respostas_por_campo": {
            # caso parte de zero — a resposta à pergunta de descoberta inicial
            # é o que dá conteúdo real pro resto do fechamento
            "objetivo": "Preciso entender por que a captação de vestibular caiu no polo de Ibmec em Brasília no último mês.",
            "tipo_demanda": "Análise",
            # ACHADO REAL (1ª rodada ao vivo, GPU_LOCAL, 07/09): este é o único
            # caso de todo o lote em que o Qwen NÃO ficou em branco a partir do
            # texto quase sem conteúdo do turno principal — ele extraiu um
            # objetivo fraco ("ajuda com dados") E JÁ ARRISCOU um tipo_demanda
            # (que não bateu com "Análise" — F1 tipo_demanda=0.0 nessa rodada),
            # tudo isso ANTES do Radio de tipo_demanda sequer aparecer (o campo
            # já não estava mais vazio quando o roteiro chegou nele). Como
            # tipo_demanda não fica mais None nesse cenário, resultado_esperado
            # PODE virar campo_prioritario_atual sem que a rota "tipo_demanda"
            # deste dict jamais seja usada (a inferência automática de Análise
            # também não roda, já que o tipo virou outra coisa) — sem esta
            # entrada de fallback, o caso trava aqui (foi exatamente o que
            # aconteceu na rodada de 07/09). Resposta abaixo é rede de
            # segurança genérica pra Produto de Dados, o cenário mais provável
            # quando isso acontece de novo. Não é bug do agente nem do
            # runner — é o próprio Qwen sendo não-determinístico com um texto
            # quase vazio; mantido como caso-alerta (ver observação abaixo).
            "resultado_esperado": "Dashboard interativo",
            "valor_negocio": "Tático",
            "classificacao_estrategica": ["Insight para Decisão"],
            "titulo": "Queda de captação do vestibular em Ibmec Brasília",
        },
        "gabarito_final": {
            "titulo": "Queda de captação do vestibular em Ibmec Brasília",
            "tipo_demanda": "Análise",
            "objetivo": "Preciso entender por que a captação de vestibular caiu no polo de Ibmec em Brasília no último mês.",
            "resultado_esperado": "Análise: Preciso entender por que a captação de vestibular caiu no polo de Ibmec em Brasília no último mês.",
            "valor_negocio": "Tático",
            "classificacao_estrategica": ["Insight para Decisão"],
            "perguntas_de_negocio": ["Por que a captação do vestibular caiu no polo de Ibmec em Brasília no último mês?"],
        },
        "observacoes": (
            "nenhum campo obrigatório é GARANTIDO extraível deste turno (por design, é o mais vago do "
            "lote) — confirma que o agente não força PRONTA nem trava quando isso acontece. Mas atenção: "
            "'nenhum campo extraível' é uma expectativa, não uma garantia — na 1ª rodada ao vivo "
            "(GPU_LOCAL, 07/09) o Qwen extraiu objetivo (fraco: 'ajuda com dados') e um tipo_demanda logo "
            "do turno principal, mesmo o texto sendo quase vazio, o que fez tipo_demanda/valor_negocio "
            "baterem F1=0.0 nessa rodada específica (gabarito espera 'Análise', o modelo decidiu outra "
            "coisa) e travou o fechamento até a resposta de resultado_esperado ser adicionada acima — ver "
            "o comentário detalhado dentro de respostas_por_campo. Não corrigir isso mudando o texto do "
            "turno pra 'blindar' contra esse comportamento — o valor deste caso É justamente expor esse "
            "tipo de reação do modelo a um pedido genérico demais; F1 baixo aqui em algumas rodadas é dado "
            "real sobre o comportamento do agente com entrada pobre, não ruído do teste. "
            "turnos_ate_pronta_esperado=7 é a estimativa do cenário 'nada extraído no turno principal' — "
            "na prática pode fechar bem mais rápido se o Qwen já vier resolvendo campos sozinho, como "
            "aconteceu na rodada de 07/09 (só não fechou daquela vez por faltar a resposta de "
            "resultado_esperado, já corrigido)."
        ),
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
