# ============================================================
# scripts/avaliar.py
# DataBrief AI — Ato 3, Bloco 26: script de avaliação de verdade
#
# Diferente de scripts/rodar_casos.py (Bloco 10/13 — bom pra revisão
# manual campo a campo e pra latência, mas sem as outras métricas),
# este script calcula as 8 métricas que o projeto pede desde o Ato 1
# (ver DataBrief_AI_Plano_4_Atos.docx e claude/ato3_plano_avaliacao.md
# — as definições de cada uma foram fechadas nessa conversa antes de
# este script existir):
#
#   1. F1 por campo (categóricos: automático; texto livre: reportado
#      lado a lado, veredito humano no Bloco 27)
#   2. Recall de lacunas (meta >=0,90)
#   3. Perguntas repetidas (meta zero — só bug de verdade)
#   4. Perguntas desnecessárias (meta <=15%)
#   5. Turnos até prontidão
#   6. Completude do briefing (meta >=0,85)
#   7. Concordância humana (meta >=80%) — mecanismo de 2 passadas
#      (Bloco 27, ver gerar_arquivo_veredito()/aplicar_veredito() mais
#      abaixo): avaliar_todos() gera o relatório com "revisao_manual"
#      (campos de texto livre lado a lado); gerar_arquivo_veredito()
#      transforma isso num CSV pra você marcar concordo/discordo linha
#      a linha; aplicar_veredito() lê o CSV preenchido, calcula o % e
#      grava um relatório final com agregado["concordancia_humana"]
#      já preenchido.
#   8. Latência P50/P95 por etapa e por modo — reaproveita
#      _resumo_latencias() de rodar_casos.py, sem reimplementar.
#
# NÃO calcula: taxa de recuperação de falhas (meta 100%) — pertence ao
# item 4 do Ato 3 (casos de falha, Bloco 29), que ainda não existe.
# O relatório final deixa essa chave como None, com uma nota.
#
# Bloco 27 — fluxo completo de concordância humana, depois de rodar
# avaliar_todos() (Célula nova, depois da Célula 3):
#
#     from scripts.avaliar import gerar_arquivo_veredito, aplicar_veredito
#
#     # 1) gera o CSV pra você preencher (uma linha por campo de texto
#     #    livre de cada caso — titulo/objetivo/bloqueios/link_evidencia,
#     #    + resultado_esperado só nos casos de Análise)
#     caminho_veredito = gerar_arquivo_veredito("resultados/avaliacoes/<arquivo>_relatorio.json")
#
#     # 2) baixe o CSV do Colab (painel de Arquivos à esquerda > "..." >
#     #    Baixar), abra no Google Sheets/Excel, preencha a coluna
#     #    "veredito" de cada linha com "concordo" ou "discordo" (a
#     #    coluna "comentario" é livre, opcional), suba de volta pro
#     #    Colab no mesmo caminho (sobrescrevendo)
#
#     # 3) lê o CSV preenchido e calcula o % de concordância
#     relatorio_final = aplicar_veredito(
#         "resultados/avaliacoes/<arquivo>_relatorio.json",
#         caminho_veredito,
#     )
#
# Bloco 25 é pré-requisito: todos os 10 casos de testes/casos_teste.py
# precisam ter respostas_por_campo completo e gabarito_final — sem
# isso, nenhum caso fecha até PRONTA e a maioria das métricas fica sem
# dado (ver achado arquitetural em claude/ato3_plano_avaliacao.md).
#
# Como rodar no Colab: igual ao rodar_casos.py — depois da Célula 3
# (grafo carregado, modelo registrado via inicializar_modelo):
#
#     from scripts.avaliar import avaliar_todos
#     relatorio = avaliar_todos()
#
# Importa deliberadamente alguns nomes "privados" (prefixo _) de
# scripts/rodar_casos.py — _valor_campo, _nova_sessao_teste,
# _percentil, _resumo_latencias, os dicts _CAMPOS_* e o limite de
# segurança. São dois scripts irmãos no mesmo pacote, testando o mesmo
# agente da mesma forma; reimplementar essa lógica aqui, já testada
# ali (Bloco 10/13/25), só criaria duas fontes de verdade pra
# divergir com o tempo.
# ============================================================

import csv
import json
import os
import time
from datetime import datetime

from schemas.models import (
    SessionState, DemandState, TipoInput, ReadinessStatus, OrigemCampo,
)
from graph.agent import (
    construir_grafo,
    processar_turno,
    processar_confirmacao_pergunta_negocio,
    processar_confirmacao_valor_negocio,
    processar_confirmacao_tipo_demanda,
    processar_confirmacao_resultado_esperado,
    processar_selecao_checkbox,
    obter_modo_ativo,
)

from testes.casos_teste import CASOS
from scripts.rodar_casos import (
    _valor_campo,
    _nova_sessao_teste,
    _rodar_turnos_principais,
    _percentil,
    _resumo_latencias,
    _LIMITE_TURNOS_FECHAMENTO,
    _CAMPOS_BARE_ENUM,
    _CAMPOS_BARE_LISTA_ENUM,
    _CAMPOS_PROVENANCE_TEXTO,
    _CAMPOS_PROVENANCE_LISTA,
)

_DIR_RESULTADOS = os.path.join("resultados", "avaliacoes")

# Os 7 campos universais que compõem o briefing — mesma lista de
# DemandState.calcular_completude(), usada aqui pra recall de lacunas
# e F1 (perguntas_de_negocio entra no recall mas NUNCA no F1, ver
# docstring do módulo e o cabeçalho de testes/casos_teste.py).
_CAMPOS_UNIVERSAIS = [
    "titulo", "tipo_demanda", "objetivo", "resultado_esperado",
    "valor_negocio", "classificacao_estrategica", "perguntas_de_negocio",
]

# Campos categóricos — comparação automática contra o gabarito entra
# no F1. Os demais (titulo, objetivo, bloqueios, link_evidencia) são
# só reportados lado a lado, pro veredito humano do Bloco 27.
_CAMPOS_CATEGORICOS = ["tipo_demanda", "valor_negocio", "classificacao_estrategica", "resultado_esperado"]
_CAMPOS_TEXTO_LIVRE_REVISAO = ["titulo", "objetivo", "bloqueios", "link_evidencia"]


def _valor_campo_ou_lista(demanda: DemandState, nome_campo: str):
    """Igual a _valor_campo, mas trata perguntas_de_negocio (lista de
    FieldProvenance) do mesmo jeito que os campos bare-lista — devolve
    lista de strings (ou lista vazia), nunca None, pra dar pra checar
    "preenchido" com um simples `bool(...)` sem caso especial em cada
    lugar que usa isso."""
    if nome_campo == "perguntas_de_negocio":
        return [fp.valor for fp in demanda.perguntas_de_negocio]
    return _valor_campo(demanda, nome_campo)


def _campos_com_falha_de_extracao(demanda_apos_principais: DemandState, campos_esperados: dict) -> set:
    """Campos que o caso dizia (`campos_esperados_apos_turnos`) que
    deveriam sair preenchidos só com os turnos principais, mas que na
    prática ficaram vazios — sinal de falha de extração do Qwen, não
    de falta de informação no texto do usuário. Usado pra marcar
    perguntas desnecessárias mais adiante (o agente pergunta de novo
    algo que o usuário já tinha dito)."""
    faltando = set()
    for campo in campos_esperados:
        atual = _valor_campo_ou_lista(demanda_apos_principais, campo)
        if not atual:
            faltando.add(campo)
    return faltando


def _lacunas_do_caso(demanda_apos_principais: DemandState, gabarito_final: dict):
    """Separa os campos universais em dois grupos, conforme a definição
    fechada em claude/ato3_plano_avaliacao.md: "lacunas" são os campos
    que o gabarito diz que a demanda final deveria ter, mas que AINDA
    não estavam preenchidos logo após os turnos principais — ou seja,
    só existem no resultado final porque foram perguntados depois.
    Campos que o gabarito não espera (valor None/vazio) nunca entram
    aqui, mesmo vazios no início — não é lacuna pedir algo que a
    demanda nunca deveria ter."""
    lacunas = []
    for campo in _CAMPOS_UNIVERSAIS:
        valor_gabarito = gabarito_final.get(campo)
        if not valor_gabarito:
            continue
        valor_apos_principais = _valor_campo_ou_lista(demanda_apos_principais, campo)
        if not valor_apos_principais:
            lacunas.append(campo)
    return lacunas


def _fechar_caso_avaliado(agente, sessao: SessionState, campo_atual: str, sugestao_pergunta,
                           respostas_por_campo: dict, campos_falha_extracao: set):
    """Mesma máquina de fechamento de scripts.rodar_casos._fechar_caso
    (mesmas 6 rotas: perguntas_de_negocio / valor_negocio / tipo_demanda
    / resultado_esperado / classificacao_estrategica / genérico), com
    instrumentação extra pras métricas 3 e 4 (perguntas repetidas e
    desnecessárias) — não dava pra reaproveitar a função original sem
    essa instrumentação por dentro do loop, então esta é uma cópia com
    os `append()` novos, não uma função nova do zero.

    Retorna (sessao, turnos_usados, motivo_parada, perguntas_repetidas,
    perguntas_desnecessarias) — as duas últimas são listas de nomes de
    campo (uma entrada por turno de pergunta em que o problema ocorreu,
    pode repetir o mesmo campo mais de uma vez)."""
    turnos_usados = 0
    perguntas_repetidas = []
    perguntas_desnecessarias = []

    while True:
        demanda = sessao.demanda_ativa
        if demanda.readiness == ReadinessStatus.PRONTA:
            return sessao, turnos_usados, None, perguntas_repetidas, perguntas_desnecessarias

        if turnos_usados >= _LIMITE_TURNOS_FECHAMENTO:
            motivo = (
                f"limite de segurança ({_LIMITE_TURNOS_FECHAMENTO} turnos) atingido sem chegar em PRONTA — "
                f"campo_prioritario_atual parado em '{campo_atual}'"
            )
            return sessao, turnos_usados, motivo, perguntas_repetidas, perguntas_desnecessarias

        # ── instrumentação (Bloco 26) — roda ANTES de responder, olhando
        # o estado ATUAL do campo que o agente está prestes a perguntar ──
        if campo_atual:
            valor_atual = _valor_campo_ou_lista(demanda, campo_atual)
            if valor_atual:
                # bug de verdade: perguntando sobre um campo que já tem valor
                perguntas_repetidas.append(campo_atual)
            if campo_atual in campos_falha_extracao:
                # o usuário já tinha dito isso no texto principal; o Qwen
                # não extraiu, então o agente precisa perguntar de novo
                perguntas_desnecessarias.append(campo_atual)

        if campo_atual == "perguntas_de_negocio":
            if not sugestao_pergunta:
                motivo = (
                    "campo_prioritario_atual == 'perguntas_de_negocio' mas nenhuma sugestão foi "
                    "gerada pelo agente — não dá pra confirmar automaticamente"
                )
                return sessao, turnos_usados, motivo, perguntas_repetidas, perguntas_desnecessarias
            sessao = processar_confirmacao_pergunta_negocio(sessao, sugestao_pergunta)
            resultado = processar_turno(agente, sessao, "__sugestao_pergunta__")
        elif campo_atual == "valor_negocio":
            if "valor_negocio" not in respostas_por_campo:
                motivo = (
                    "campo_prioritario_atual == 'valor_negocio' mas não há resposta prevista "
                    "em respostas_por_campo — ajuste o caso de teste"
                )
                return sessao, turnos_usados, motivo, perguntas_repetidas, perguntas_desnecessarias
            sessao = processar_confirmacao_valor_negocio(sessao, respostas_por_campo["valor_negocio"])
            resultado = processar_turno(agente, sessao, "__radio__")
        elif campo_atual == "tipo_demanda":
            if "tipo_demanda" not in respostas_por_campo:
                motivo = (
                    "campo_prioritario_atual == 'tipo_demanda' mas não há resposta prevista "
                    "em respostas_por_campo — ajuste o caso de teste"
                )
                return sessao, turnos_usados, motivo, perguntas_repetidas, perguntas_desnecessarias
            sessao = processar_confirmacao_tipo_demanda(sessao, respostas_por_campo["tipo_demanda"])
            resultado = processar_turno(agente, sessao, "__radio__")
        elif campo_atual == "resultado_esperado":
            if "resultado_esperado" not in respostas_por_campo:
                motivo = (
                    "campo_prioritario_atual == 'resultado_esperado' mas não há resposta prevista "
                    "em respostas_por_campo — ajuste o caso de teste"
                )
                return sessao, turnos_usados, motivo, perguntas_repetidas, perguntas_desnecessarias
            sessao = processar_confirmacao_resultado_esperado(sessao, respostas_por_campo["resultado_esperado"])
            resultado = processar_turno(agente, sessao, "__radio__")
        elif campo_atual == "classificacao_estrategica":
            if "classificacao_estrategica" not in respostas_por_campo:
                motivo = (
                    "campo_prioritario_atual == 'classificacao_estrategica' mas não há resposta "
                    "prevista em respostas_por_campo — ajuste o caso de teste"
                )
                return sessao, turnos_usados, motivo, perguntas_repetidas, perguntas_desnecessarias
            valor = respostas_por_campo["classificacao_estrategica"]
            valores = valor if isinstance(valor, list) else [valor]
            sessao = processar_selecao_checkbox(sessao, valores)
            resultado = processar_turno(agente, sessao, "__checkbox__")
        elif campo_atual in respostas_por_campo:
            resultado = processar_turno(agente, sessao, respostas_por_campo[campo_atual])
        else:
            motivo = (
                f"sem resposta prevista pro campo '{campo_atual}' em respostas_por_campo — "
                f"ajuste o caso de teste"
            )
            return sessao, turnos_usados, motivo, perguntas_repetidas, perguntas_desnecessarias

        sessao, _, _, campo_atual, sugestao_pergunta = resultado
        turnos_usados += 1


def _resultado_esperado_correto(gabarito_tipo_demanda: str, gabarito_resultado, resultado_obtido) -> bool:
    """resultado_esperado tem uma regra especial (ver claude/ato3_plano_avaliacao.md):
    quando tipo_demanda == Análise, o valor de verdade é "Análise: {objetivo}" —
    o texto do objetivo embutido varia por caso e não é seguro comparar como
    string exata (é texto livre, entra na revisão humana igual aos outros).
    Aqui só validamos a ESTRUTURA: prefixo "Análise: " presente e origem RULE
    (a inferência automática de no_avaliar_completude, nunca resposta manual).
    Pra Produto de Dados/Alarmística, resultado_esperado é 100% categórico
    (um de 4 valores fixos) — comparação exata mesmo."""
    if resultado_obtido is None:
        return False
    if gabarito_tipo_demanda == "Análise":
        return resultado_obtido.valor.startswith("Análise: ") and resultado_obtido.origem == OrigemCampo.RULE
    return resultado_obtido.valor == gabarito_resultado


def _f1_conjunto(gabarito_lista: list, obtido_lista: list) -> float:
    """F1 sobre interseção de conjuntos — usado só pra classificacao_estrategica,
    o único campo categórico que pode ter mais de um valor certo ao mesmo tempo."""
    gabarito = set(gabarito_lista or [])
    obtido = set(obtido_lista or [])
    if not gabarito and not obtido:
        return 1.0
    intersecao = len(gabarito & obtido)
    if intersecao == 0:
        return 0.0
    precisao = intersecao / len(obtido) if obtido else 0.0
    recall = intersecao / len(gabarito) if gabarito else 0.0
    if precisao + recall == 0:
        return 0.0
    return 2 * precisao * recall / (precisao + recall)


def avaliar_caso(agente, caso: dict) -> dict:
    """Roda um caso até PRONTA (ou até parar) e calcula todas as métricas
    automáticas sobre ele. Não derruba o lote se um caso falhar — registra
    o erro e segue (mesmo padrão de scripts.rodar_casos.rodar_caso)."""
    cid = caso["id"]
    print(f"\n{'=' * 60}")
    print(f"{cid} — {caso['categoria']}")
    print("=" * 60)

    t_inicio = time.time()

    try:
        gabarito_final = caso.get("gabarito_final")
        respostas_por_campo = caso.get("respostas_por_campo", {})

        if not gabarito_final:
            print(f"  ⚠️  {cid} não tem gabarito_final — pulando (não dá pra avaliar sem gabarito)")
            return {
                "id": cid, "erro": None, "pulado": True,
                "motivo_pulado": "sem gabarito_final",
                "log_latencias": {},
            }

        sessao = _nova_sessao_teste()
        sessao, campo_atual, sugestao_pergunta = _rodar_turnos_principais(agente, sessao, caso["turnos"])

        # snapshot ANTES do fechamento dinâmico — precisa ser cópia de
        # verdade (Pydantic .model_copy(deep=True)), porque as funções de
        # confirmação e processar_turno mutam o MESMO objeto DemandState
        # dentro de sessao.demandas — sem cópia, "antes" e "depois" seriam
        # o objeto idêntico por referência.
        demanda_apos_principais = sessao.demanda_ativa.model_copy(deep=True)

        campos_falha_extracao = _campos_com_falha_de_extracao(
            demanda_apos_principais, caso["campos_esperados_apos_turnos"]
        )

        sessao, turnos_extra, motivo_parada, perguntas_repetidas, perguntas_desnecessarias = _fechar_caso_avaliado(
            agente, sessao, campo_atual, sugestao_pergunta, respostas_por_campo, campos_falha_extracao
        )
        demanda_final = sessao.demanda_ativa
        chegou_pronta = motivo_parada is None
        turnos_ate_pronta = (len(caso["turnos"]) + turnos_extra) if chegou_pronta else None

        if chegou_pronta:
            print(f"  ✅ PRONTA em {turnos_ate_pronta} turnos (estimativa: {caso.get('turnos_ate_pronta_esperado')})")
        else:
            print(f"  ⚠️  não chegou em PRONTA: {motivo_parada}")

        # ── 1. F1 por campo (categóricos) ──
        f1_por_campo = {}
        for campo in _CAMPOS_CATEGORICOS:
            valor_gabarito = gabarito_final.get(campo)
            if valor_gabarito is None:
                continue  # gabarito não define esse campo pra este caso — não entra na média
            if campo == "classificacao_estrategica":
                obtido = _valor_campo(demanda_final, campo)
                f1_por_campo[campo] = _f1_conjunto(valor_gabarito, obtido)
            elif campo == "resultado_esperado":
                correto = _resultado_esperado_correto(
                    gabarito_final.get("tipo_demanda"), valor_gabarito, demanda_final.resultado_esperado
                )
                f1_por_campo[campo] = 1.0 if correto else 0.0
            else:
                obtido = _valor_campo(demanda_final, campo)
                f1_por_campo[campo] = 1.0 if obtido == valor_gabarito else 0.0

        # ── 2. Recall de lacunas ──
        lacunas = _lacunas_do_caso(demanda_apos_principais, gabarito_final)
        lacunas_resolvidas = [c for c in lacunas if _valor_campo_ou_lista(demanda_final, c)]
        recall_lacunas = (len(lacunas_resolvidas) / len(lacunas)) if lacunas else None

        # ── 3/4. perguntas repetidas/desnecessárias — já vieram do fechamento ──
        total_perguntas = turnos_extra

        # ── 6. completude do briefing ──
        completude = demanda_final.calcular_completude()

        # ── revisão manual (texto livre) — meia metade da concordância
        # humana (Bloco 27 lê o resto) ──
        revisao_manual = []
        for campo in _CAMPOS_TEXTO_LIVRE_REVISAO:
            valor_gabarito = gabarito_final.get(campo)
            if valor_gabarito is None:
                continue
            obtido = _valor_campo(demanda_final, campo)
            revisao_manual.append({
                "chave": f"{cid}:{campo}",
                "campo": campo,
                "gabarito": valor_gabarito,
                "obtido": obtido,
            })
        # resultado_esperado de Análise também precisa de revisão (o texto
        # do objetivo embutido, não a estrutura — essa parte já foi
        # validada automaticamente acima)
        if gabarito_final.get("tipo_demanda") == "Análise" and demanda_final.resultado_esperado:
            revisao_manual.append({
                "chave": f"{cid}:resultado_esperado",
                "campo": "resultado_esperado",
                "gabarito": gabarito_final.get("resultado_esperado"),
                "obtido": demanda_final.resultado_esperado.valor,
            })

        duracao = time.time() - t_inicio
        log_latencias = dict(demanda_final.log_latencias)

        return {
            "id": cid,
            "erro": None,
            "pulado": False,
            "chegou_pronta": chegou_pronta,
            "motivo_nao_pronta": motivo_parada,
            "turnos_ate_pronta": turnos_ate_pronta,
            "turnos_ate_pronta_esperado": caso.get("turnos_ate_pronta_esperado"),
            "f1_por_campo": f1_por_campo,
            "lacunas": lacunas,
            "lacunas_resolvidas": lacunas_resolvidas,
            "recall_lacunas": recall_lacunas,
            "perguntas_repetidas": perguntas_repetidas,
            "perguntas_desnecessarias": perguntas_desnecessarias,
            "total_perguntas": total_perguntas,
            "completude": completude,
            "tentativas_pergunta": dict(demanda_final.tentativas_pergunta),
            "campos_desistidos": list(demanda_final.campos_desistidos),
            "revisao_manual": revisao_manual,
            "duracao_caso_s": round(duracao, 2),
            "log_latencias": log_latencias,
        }

    except Exception as e:
        duracao = time.time() - t_inicio
        print(f"  💥 ERRO ao avaliar {cid}: {type(e).__name__}: {e}")
        return {
            "id": cid, "erro": str(e), "pulado": False,
            "log_latencias": {}, "duracao_caso_s": round(duracao, 2),
        }


def _agregar_metricas(relatorios: list) -> dict:
    """Junta as métricas de todos os casos num resumo único — médias,
    somas e percentuais, prontos pra comparar contra as metas do
    projeto (ver docstring do módulo)."""
    validos = [r for r in relatorios if not r.get("erro") and not r.get("pulado")]

    # F1 por campo — média por campo, só sobre casos que definiam gabarito
    # pra aquele campo (nem todo caso testa todos os campos categóricos)
    f1_agregado = {}
    for campo in _CAMPOS_CATEGORICOS:
        valores = [r["f1_por_campo"][campo] for r in validos if campo in r.get("f1_por_campo", {})]
        if valores:
            f1_agregado[campo] = round(sum(valores) / len(valores), 3)

    # Recall de lacunas — agregado GLOBAL (soma de lacunas resolvidas sobre
    # soma de lacunas totais, não média das médias — um caso com 1 lacuna
    # não deveria pesar igual a um com 5)
    total_lacunas = sum(len(r["lacunas"]) for r in validos)
    total_resolvidas = sum(len(r["lacunas_resolvidas"]) for r in validos)
    recall_lacunas_global = (total_resolvidas / total_lacunas) if total_lacunas else None

    # Perguntas repetidas/desnecessárias — contagem total + % sobre o total
    # de perguntas feitas em todo o lote
    total_perguntas_feitas = sum(r.get("total_perguntas", 0) for r in validos)
    total_repetidas = sum(len(r.get("perguntas_repetidas", [])) for r in validos)
    total_desnecessarias = sum(len(r.get("perguntas_desnecessarias", [])) for r in validos)
    pct_desnecessarias = (total_desnecessarias / total_perguntas_feitas) if total_perguntas_feitas else None

    # Turnos até prontidão — só sobre os que chegaram em PRONTA
    turnos = [r["turnos_ate_pronta"] for r in validos if r.get("chegou_pronta")]

    # Completude — média sobre todos os casos válidos (não só os prontos —
    # completude mede o estado final, mesmo que tenha parado antes)
    completudes = [r["completude"] for r in validos if "completude" in r]

    # tentativas por campo — média sobre TODAS as entradas de
    # tentativas_pergunta de TODOS os casos (não só campos que travaram)
    todas_tentativas = []
    for r in validos:
        todas_tentativas.extend(r.get("tentativas_pergunta", {}).values())
    total_desistidos = sum(len(r.get("campos_desistidos", [])) for r in validos)

    return {
        "n_casos": len(relatorios),
        "n_avaliados": len(validos),
        "n_erros": len([r for r in relatorios if r.get("erro")]),
        "n_pulados": len([r for r in relatorios if r.get("pulado")]),
        "n_chegaram_pronta": len([r for r in validos if r.get("chegou_pronta")]),
        "f1_por_campo": f1_agregado,
        "recall_lacunas_global": round(recall_lacunas_global, 3) if recall_lacunas_global is not None else None,
        "meta_recall_lacunas": 0.90,
        "perguntas_repetidas_total": total_repetidas,
        "meta_perguntas_repetidas": 0,
        "perguntas_desnecessarias_total": total_desnecessarias,
        "pct_perguntas_desnecessarias": round(pct_desnecessarias, 3) if pct_desnecessarias is not None else None,
        "meta_pct_perguntas_desnecessarias": 0.15,
        "turnos_ate_pronta_media": round(sum(turnos) / len(turnos), 2) if turnos else None,
        "turnos_ate_pronta_mediana": round(_percentil(turnos, 50), 2) if turnos else None,
        "completude_media": round(sum(completudes) / len(completudes), 3) if completudes else None,
        "meta_completude": 0.85,
        "tentativas_pergunta_media": round(sum(todas_tentativas) / len(todas_tentativas), 2) if todas_tentativas else None,
        "campos_desistidos_total": total_desistidos,
        "concordancia_humana": None,  # Bloco 27 — depende do arquivo de veredito
        "meta_concordancia_humana": 0.80,
        "taxa_recuperacao_falhas": None,  # Bloco 29 — depende dos casos de falha
        "meta_taxa_recuperacao_falhas": 1.00,
    }


def avaliar_todos(casos: list = None, salvar_json: bool = True) -> dict:
    """Roda avaliar_caso() em todos os casos, agrega as métricas e grava
    um relatório completo em resultados/avaliacoes/. É o script de
    avaliação de verdade do Ato 3 item 3 — NÃO substitui
    scripts.rodar_casos.rodar_todos() (que continua servindo pra
    latência comparada entre modos, Bloco 13); este aqui é sobre
    qualidade da extração/fechamento, não desempenho."""
    casos = casos if casos is not None else CASOS
    modo = obter_modo_ativo()
    print(f"Modo de execução ativo: {modo.value}")
    print(f"Avaliando {len(casos)} caso(s)...")

    agente = construir_grafo()

    t_inicio = time.time()
    relatorios = [avaliar_caso(agente, caso) for caso in casos]
    duracao_lote = time.time() - t_inicio

    agregado = _agregar_metricas(relatorios)
    resumo_latencias = _resumo_latencias(relatorios)

    print(f"\n{'=' * 60}")
    print("RESUMO DA AVALIAÇÃO")
    print("=" * 60)
    print(f"Modo: {modo.value}")
    print(f"{agregado['n_casos']} casos, {agregado['n_avaliados']} avaliados, "
          f"{agregado['n_erros']} com erro, {agregado['n_pulados']} pulados (sem gabarito), "
          f"{agregado['n_chegaram_pronta']} chegaram em PRONTA.")
    print(f"\nF1 por campo: {agregado['f1_por_campo']}")
    print(f"Recall de lacunas: {agregado['recall_lacunas_global']} (meta >= {agregado['meta_recall_lacunas']})")
    print(f"Perguntas repetidas: {agregado['perguntas_repetidas_total']} (meta = {agregado['meta_perguntas_repetidas']})")
    print(f"Perguntas desnecessárias: {agregado['perguntas_desnecessarias_total']} "
          f"({agregado['pct_perguntas_desnecessarias']} — meta <= {agregado['meta_pct_perguntas_desnecessarias']})")
    print(f"Turnos até prontidão: média={agregado['turnos_ate_pronta_media']} "
          f"mediana={agregado['turnos_ate_pronta_mediana']}")
    print(f"Completude do briefing: média={agregado['completude_media']} (meta >= {agregado['meta_completude']})")
    print(f"Concordância humana: {agregado['concordancia_humana']} — pendente (Bloco 27)")
    print(f"Taxa de recuperação de falhas: {agregado['taxa_recuperacao_falhas']} — pendente (Bloco 29)")

    print("\nCasos que precisam de revisão manual (texto livre + concordância humana):")
    total_revisao = sum(len(r.get("revisao_manual", [])) for r in relatorios if not r.get("erro"))
    print(f"  {total_revisao} item(ns) — ver 'revisao_manual' em cada caso do relatório salvo.")

    if salvar_json:
        os.makedirs(_DIR_RESULTADOS, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        caminho = os.path.join(_DIR_RESULTADOS, f"{modo.value}_{timestamp}_relatorio.json")
        conteudo = {
            "modo": modo.value,
            "timestamp": timestamp,
            "duracao_lote_s": round(duracao_lote, 2),
            "agregado": agregado,
            "resumo_latencias": resumo_latencias,
            "casos": relatorios,
        }
        with open(caminho, "w", encoding="utf-8") as f:
            json.dump(conteudo, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n💾 relatório salvo em {caminho} — commite esse arquivo. O Bloco 27 vai ler os "
              f"itens de 'revisao_manual' de dentro dele pra montar o arquivo de veredito.")

    return {
        "modo": modo.value,
        "agregado": agregado,
        "resumo_latencias": resumo_latencias,
        "casos": relatorios,
    }


# ────────────────────────────────────────────────────────────
# Bloco 27 — mecanismo de veredito humano (concordância humana, meta >=80%)
#
# Duas funções, 2 passadas — ver exemplo de uso completo no cabeçalho
# do módulo. CSV escolhido em vez de JSON pro arquivo de veredito
# porque é bem mais rápido de preencher numa planilha (Google Sheets/
# Excel) do que editando texto/aspas/vírgulas à mão num JSON — só uma
# palavra por linha ("concordo" ou "discordo"), sem risco de quebrar a
# sintaxe do arquivo.
# ────────────────────────────────────────────────────────────

_VEREDITOS_VALIDOS = {"concordo", "discordo"}
_COLUNAS_VEREDITO = ["chave", "caso_id", "campo", "gabarito", "obtido", "veredito", "comentario"]


def gerar_arquivo_veredito(caminho_relatorio: str, caminho_saida: str = None) -> str:
    """1ª passada do Bloco 27. Lê um relatório já salvo por avaliar_todos()
    e monta um CSV com uma linha por item de 'revisao_manual' de cada caso
    (titulo/objetivo/bloqueios/link_evidencia sempre, resultado_esperado só
    nos casos de Análise — ver avaliar_caso()), pronto pra você preencher a
    coluna 'veredito' com 'concordo' ou 'discordo'.

    Se caminho_saida já existir (por exemplo, você rodou isso antes numa
    rodada anterior do relatório), os vereditos e comentários já
    preenchidos são PRESERVADOS — só entram linhas novas pra itens que
    ainda não tinham sido julgados. Rodar de novo depois de preencher tudo
    não apaga nada.

    Retorna o caminho do CSV gerado (mesmo nome do relatório, trocando
    "_relatorio.json" por "_veredito.csv", se caminho_saida não for
    informado)."""
    with open(caminho_relatorio, encoding="utf-8") as f:
        relatorio = json.load(f)

    if caminho_saida is None:
        if caminho_relatorio.endswith("_relatorio.json"):
            caminho_saida = caminho_relatorio[: -len("_relatorio.json")] + "_veredito.csv"
        else:
            caminho_saida = caminho_relatorio + ".veredito.csv"

    vereditos_existentes = {}
    if os.path.exists(caminho_saida):
        with open(caminho_saida, encoding="utf-8", newline="") as f:
            for linha in csv.DictReader(f):
                vereditos_existentes[linha["chave"]] = (
                    linha.get("veredito", ""), linha.get("comentario", ""),
                )

    linhas = []
    for caso in relatorio.get("casos", []):
        for item in caso.get("revisao_manual", []):
            veredito_previo, comentario_previo = vereditos_existentes.get(item["chave"], ("", ""))
            linhas.append({
                "chave": item["chave"],
                "caso_id": caso["id"],
                "campo": item["campo"],
                "gabarito": item["gabarito"],
                "obtido": item["obtido"],
                "veredito": veredito_previo,
                "comentario": comentario_previo,
            })

    with open(caminho_saida, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_COLUNAS_VEREDITO)
        writer.writeheader()
        writer.writerows(linhas)

    n_pendentes = sum(
        1 for l in linhas if l["veredito"].strip().lower() not in _VEREDITOS_VALIDOS
    )
    print(f"📝 arquivo de veredito gerado/atualizado em {caminho_saida}")
    print(f"   {len(linhas)} item(ns) no total, {n_pendentes} pendente(s) de julgamento.")
    print("   Preencha a coluna 'veredito' de cada linha com 'concordo' ou 'discordo' "
          "('comentario' é livre, opcional) e rode aplicar_veredito() depois.")
    return caminho_saida


def aplicar_veredito(caminho_relatorio: str, caminho_veredito: str, salvar_json: bool = True) -> dict:
    """2ª passada do Bloco 27. Lê o CSV de veredito já preenchido, calcula
    a concordância humana (concordo / total JULGADO — itens sem veredito
    válido preenchido ficam de fora da conta, listados como pendentes ou
    inválidos em vez de contar como discordância), atualiza
    agregado['concordancia_humana'] e grava um relatório final (arquivo
    novo, nunca sobrescreve o relatório original de avaliar_todos()).

    Uma linha com 'veredito' vazio conta como pendente (ainda não julgada).
    Uma linha com qualquer outro texto que não seja exatamente 'concordo'
    ou 'discordo' (fora de maiúsculas/espaços) conta como inválida — os
    dois casos ficam de fora do cálculo e são listados no relatório pra
    você corrigir, em vez de distorcer o percentual silenciosamente."""
    with open(caminho_relatorio, encoding="utf-8") as f:
        relatorio = json.load(f)

    vereditos = {}
    with open(caminho_veredito, encoding="utf-8", newline="") as f:
        for linha in csv.DictReader(f):
            vereditos[linha["chave"]] = linha.get("veredito", "")

    concordo = 0
    discordo = 0
    pendentes = []
    invalidos = []
    total_itens = 0

    for caso in relatorio.get("casos", []):
        for item in caso.get("revisao_manual", []):
            chave = item["chave"]
            total_itens += 1
            valor_bruto = vereditos.get(chave, "")
            valor = valor_bruto.strip().lower()
            if valor == "concordo":
                concordo += 1
            elif valor == "discordo":
                discordo += 1
            elif valor == "":
                pendentes.append(chave)
            else:
                invalidos.append({"chave": chave, "valor": valor_bruto})

    total_julgado = concordo + discordo
    pct_concordancia = (concordo / total_julgado) if total_julgado else None

    relatorio["agregado"]["concordancia_humana"] = (
        round(pct_concordancia, 3) if pct_concordancia is not None else None
    )
    relatorio["veredito"] = {
        "arquivo": caminho_veredito,
        "concordo": concordo,
        "discordo": discordo,
        "total_julgado": total_julgado,
        "total_itens": total_itens,
        "pendentes": pendentes,
        "invalidos": invalidos,
    }

    meta = relatorio["agregado"].get("meta_concordancia_humana", 0.80)
    print(f"Concordância humana: {relatorio['agregado']['concordancia_humana']} (meta >= {meta}) "
          f"— {concordo} concordo / {discordo} discordo / {len(pendentes)} pendente(s) de {total_itens}")
    if invalidos:
        print(f"⚠️  {len(invalidos)} linha(s) com 'veredito' não reconhecido (esperado "
              f"'concordo' ou 'discordo'), fora da conta: {invalidos}")
    if pendentes:
        print(f"⚠️  {len(pendentes)} item(ns) ainda sem veredito, fora da conta: {pendentes}")

    if salvar_json:
        if caminho_relatorio.endswith("_relatorio.json"):
            caminho_final = caminho_relatorio[: -len("_relatorio.json")] + "_final.json"
        else:
            caminho_final = caminho_relatorio + ".final.json"
        with open(caminho_final, "w", encoding="utf-8") as f:
            json.dump(relatorio, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n💾 relatório final salvo em {caminho_final} — commite esse arquivo junto "
              f"com o CSV de veredito.")

    return relatorio


if __name__ == "__main__":
    avaliar_todos()
