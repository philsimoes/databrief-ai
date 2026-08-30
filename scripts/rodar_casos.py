# ============================================================
# scripts/rodar_casos.py
# DataBrief AI — Ato 3, Bloco 13: runner manual dos casos de teste
# (com captura de latência pra comparação entre modos)
#
# NÃO é o script de avaliação completo do item 3 do Ato 3 (F1 por
# campo, recall de lacunas, concordância humana etc. — isso ainda não
# foi construído). Isso aqui é um runner enxuto pra rodar os casos
# de `testes/casos_teste.py` contra o agente já carregado na sessão
# do Colab e mostrar o que foi extraído campo a campo, pra revisão
# manual (a mesma lógica de "concordância humana" que o projeto já
# pede como métrica, só que feita à mão por enquanto).
#
# Bloco 13 (novo): cada execução agora captura tempo total por caso
# e reaproveita demanda.log_latencias (já existente em graph/agent.py,
# preenchido por chamar_llm em cada etapa — extração/pergunta/briefing)
# pra montar um resumo de latência por etapa e por modo. No final,
# grava um JSON em resultados/execucoes/ com tudo isso, já rotulado
# com o modo de execução ativo (CPU_LOCAL / GPU_LOCAL / OPENAI) e um
# timestamp — pra depois dar pra comparar dois arquivos (um de cada
# modo) numericamente, em vez de só lembrar "essa rodada pareceu mais
# rápida". Isso adianta parte do item 3 do Ato 3 (latência P50/P95 por
# etapa e por modo), mas ainda não é o script de avaliação final.
#
# Como rodar no Colab: depois da Célula 3 (grafo carregado e modelo
# registrado via inicializar_modelo) — NÃO precisa da Célula 4/Gradio
# rodando — abra uma célula nova e rode:
#
#     from scripts.rodar_casos import rodar_todos
#     rodar_todos()
#
# Rodar como `!python scripts/rodar_casos.py` NÃO funciona — isso
# abriria um processo Python novo, sem o modelo carregado. Tem que
# ser import + chamada de função, na mesma sessão do kernel.
# ============================================================

import json
import os
import time
from datetime import datetime

from schemas.models import SessionState, DemandState, TipoInput, ReadinessStatus
from graph.agent import (
    construir_grafo,
    processar_turno,
    processar_confirmacao_pergunta_negocio,
    obter_modo_ativo,
)

from testes.casos_teste import CASOS

# Segurança contra loop: nenhum caso fechado deveria precisar de mais
# turnos que isso pra chegar em PRONTA — se estourar, é sinal de bug
# (campo sem resposta prevista, ou o agente pedindo o mesmo campo de novo)
_LIMITE_TURNOS_FECHAMENTO = 8

# Campos que NÃO são wrapeados em FieldProvenance no DemandState —
# valor direto do enum (confirmado em schemas/models.py)
_CAMPOS_BARE_ENUM = {"tipo_demanda", "valor_negocio"}
_CAMPOS_BARE_LISTA_ENUM = {"classificacao_estrategica"}
# Campos wrapeados em FieldProvenance — precisa pegar .valor
_CAMPOS_PROVENANCE_TEXTO = {"titulo", "objetivo", "resultado_esperado", "bloqueios", "link_evidencia"}
_CAMPOS_PROVENANCE_LISTA = {"perguntas_de_negocio"}

_DIR_RESULTADOS = os.path.join("resultados", "execucoes")


def _valor_campo(demanda: DemandState, nome_campo: str):
    """Extrai o valor comparável de um campo, sabendo se ele é bare ou
    wrapeado em FieldProvenance — essa distinção é uma inconsistência real
    do schema (ver claude/ato3_kickoff.md), não um detalhe do runner."""
    bruto = getattr(demanda, nome_campo, None)
    if bruto is None:
        return None
    if nome_campo in _CAMPOS_BARE_ENUM:
        return bruto.value
    if nome_campo in _CAMPOS_BARE_LISTA_ENUM:
        return [item.value for item in bruto]
    if nome_campo in _CAMPOS_PROVENANCE_TEXTO:
        return bruto.valor
    if nome_campo in _CAMPOS_PROVENANCE_LISTA:
        return [fp.valor for fp in bruto]
    return bruto


def _percentil(valores: list, p: float) -> float:
    """Percentil por interpolação linear, sem depender de numpy (não dá
    pra garantir que tá instalado em toda sessão do Colab)."""
    if not valores:
        return 0.0
    ordenado = sorted(valores)
    if len(ordenado) == 1:
        return ordenado[0]
    k = (len(ordenado) - 1) * (p / 100)
    piso = int(k)
    teto = min(piso + 1, len(ordenado) - 1)
    if piso == teto:
        return ordenado[piso]
    return ordenado[piso] + (ordenado[teto] - ordenado[piso]) * (k - piso)


def _nova_sessao_teste() -> SessionState:
    """Sessão isolada por caso — mesmo padrão de nova_sessao() do
    interface/app.py, sem depender da UI."""
    sessao = SessionState(modo_execucao=obter_modo_ativo())
    sessao.adicionar_demanda(DemandState())
    return sessao


def _rodar_turnos_principais(agente, sessao: SessionState, turnos: list):
    """Roda a lista fixa de turnos principais do caso. Retorna
    (sessao, campo_prioritario_atual, sugestao_pergunta_negocio) — os dois
    últimos são usados só pelos casos fechados, pra saber por onde
    continuar (sugestao_pergunta_negocio só vem preenchida quando o
    próprio campo_atual já é "perguntas_de_negocio" logo após o turno
    principal — não é um campo do schema, é o 5º valor que processar_turno
    devolve)."""
    campo_atual = ""
    sugestao_pergunta = None
    for turno in turnos:
        resultado = processar_turno(
            agente,
            sessao,
            turno["conteudo"],
            tipo=TipoInput(turno["tipo"]),
            nome_arquivo=turno.get("nome_arquivo"),
        )
        sessao, _, _, campo_atual, sugestao_pergunta = resultado
    return sessao, campo_atual, sugestao_pergunta


def _fechar_caso(agente, sessao: SessionState, campo_atual: str, sugestao_pergunta,
                  respostas_por_campo: dict):
    """Continua o roteiro dinamicamente até PRONTA (ou até o limite de
    segurança), usando respostas_por_campo pros campos com resposta
    canned e o fluxo real de confirmação (não texto livre) para
    perguntas_de_negocio — esse campo nunca é extraído do texto do
    usuário (ver aplicar_extracao em graph/agent.py). `sugestao_pergunta`
    é passada como variável local entre iterações (não fica guardada no
    objeto SessionState — é Pydantic, não aceita atributo extra não
    declarado no schema).

    Retorna (sessao, turnos_usados, motivo_parada) — motivo_parada é
    None se chegou em PRONTA, ou uma string explicando por que parou
    antes disso (sem resposta prevista, ou limite de segurança).
    """
    turnos_usados = 0
    while True:
        demanda = sessao.demanda_ativa
        if demanda.readiness == ReadinessStatus.PRONTA:
            return sessao, turnos_usados, None

        if turnos_usados >= _LIMITE_TURNOS_FECHAMENTO:
            return sessao, turnos_usados, (
                f"limite de segurança ({_LIMITE_TURNOS_FECHAMENTO} turnos) atingido sem chegar em PRONTA — "
                f"campo_prioritario_atual parado em '{campo_atual}'"
            )

        if campo_atual == "perguntas_de_negocio":
            # Fluxo real: aceita a própria sugestão que o Qwen gerou (o
            # 5º valor do processar_turno anterior) — mesmo comportamento
            # de um usuário que só confirma sem editar.
            if not sugestao_pergunta:
                return sessao, turnos_usados, (
                    "campo_prioritario_atual == 'perguntas_de_negocio' mas nenhuma sugestão foi "
                    "gerada pelo agente — não dá pra confirmar automaticamente"
                )
            sessao = processar_confirmacao_pergunta_negocio(sessao, sugestao_pergunta)
            resultado = processar_turno(agente, sessao, "__sugestao_pergunta__")
        elif campo_atual in respostas_por_campo:
            resultado = processar_turno(agente, sessao, respostas_por_campo[campo_atual])
        else:
            return sessao, turnos_usados, (
                f"sem resposta prevista pro campo '{campo_atual}' em respostas_por_campo — "
                f"ajuste o caso de teste"
            )

        sessao, _, _, campo_atual, sugestao_pergunta = resultado
        turnos_usados += 1


def rodar_caso(agente, caso: dict) -> dict:
    """Roda um caso (turnos principais + fechamento dinâmico, se houver)
    e monta um relatório campo a campo pra revisão manual. Não derruba o
    lote inteiro se um caso der erro — registra e segue pro próximo.

    Também mede o tempo total do caso (wall-clock) e recolhe
    demanda.log_latencias (já existente em graph/agent.py, uma entrada por
    chamada ao LLM) pra permitir comparação de desempenho entre modos."""
    print(f"\n{'=' * 60}")
    print(f"{caso['id']} — {caso['categoria']}")
    print(f"  {caso['descricao']}")
    print("=" * 60)

    t_inicio_caso = time.time()

    try:
        sessao = _nova_sessao_teste()
        sessao, campo_atual, sugestao_pergunta = _rodar_turnos_principais(agente, sessao, caso["turnos"])
        demanda = sessao.demanda_ativa

        print(f"\n  readiness após turnos principais: {demanda.readiness.value}"
              f"  (esperado: {caso['readiness_esperado_apos_turnos']})")

        for nome_campo, esperado in caso["campos_esperados_apos_turnos"].items():
            atual = _valor_campo(demanda, nome_campo)
            if nome_campo in _CAMPOS_BARE_ENUM:
                status = "✅" if atual == esperado else "❌"
            elif nome_campo in _CAMPOS_BARE_LISTA_ENUM:
                faltando = set(esperado) - set(atual or [])
                status = "✅" if not faltando else f"❌ faltando: {faltando}"
            else:
                # campo de texto livre — comparação exata não é realista
                # com um LLM, isso aqui é só uma dica visual pra revisão
                bate = bool(atual) and (
                    esperado.lower() in atual.lower() or atual.lower() in esperado.lower()
                )
                status = "🔍 revisar" if not bate else "✅ (aparenta bater)"
            print(f"    {nome_campo}: {status}\n      esperado: {esperado!r}\n      obtido:   {atual!r}")

        # ── caso fechado: continua dinamicamente até PRONTA ──
        respostas_por_campo = caso.get("respostas_por_campo")
        motivo_parada = None
        if respostas_por_campo:
            sessao, turnos_extra, motivo_parada = _fechar_caso(
                agente, sessao, campo_atual, sugestao_pergunta, respostas_por_campo
            )
            demanda = sessao.demanda_ativa
            turno_final = len(caso["turnos"]) + turnos_extra

            if motivo_parada:
                print(f"\n  ⚠️  não chegou em PRONTA: {motivo_parada}")
                print(f"  pendências restantes: {demanda.pendencias}")
            else:
                print(f"\n  ✅ PRONTA em {turno_final} turnos "
                      f"(estimativa do caso: {caso.get('turnos_ate_pronta_esperado')})")

        duracao_caso = time.time() - t_inicio_caso
        log_latencias = dict(demanda.log_latencias)
        n_chamadas = len(log_latencias)
        print(f"\n  ⏱  tempo total do caso: {duracao_caso:.1f}s "
              f"({n_chamadas} chamada(s) ao LLM, "
              f"média {duracao_caso / n_chamadas if n_chamadas else 0:.1f}s/chamada)")

        return {
            "id": caso["id"],
            "erro": None,
            "duracao_caso_s": round(duracao_caso, 2),
            "log_latencias": log_latencias,
            "motivo_nao_pronta": motivo_parada,
        }

    except Exception as e:
        duracao_caso = time.time() - t_inicio_caso
        print(f"\n  💥 ERRO ao rodar este caso: {type(e).__name__}: {e}")
        return {
            "id": caso["id"],
            "erro": str(e),
            "duracao_caso_s": round(duracao_caso, 2),
            "log_latencias": {},
            "motivo_nao_pronta": None,
        }


def _resumo_latencias(relatorios: list) -> dict:
    """Agrupa log_latencias de todos os casos por etapa (extracao/pergunta/
    briefing — o prefixo antes de '_turno_N' em cada chave) e calcula
    contagem, média, P50 e P95 de cada grupo, além do agregado geral."""
    por_etapa = {}
    for r in relatorios:
        for chave, valor in r["log_latencias"].items():
            etapa = chave.split("_turno_")[0]
            por_etapa.setdefault(etapa, []).append(valor)

    resumo = {}
    todos_valores = []
    for etapa, valores in por_etapa.items():
        todos_valores.extend(valores)
        resumo[etapa] = {
            "n": len(valores),
            "media_s": round(sum(valores) / len(valores), 2),
            "p50_s": round(_percentil(valores, 50), 2),
            "p95_s": round(_percentil(valores, 95), 2),
        }
    resumo["_geral"] = {
        "n": len(todos_valores),
        "media_s": round(sum(todos_valores) / len(todos_valores), 2) if todos_valores else 0.0,
        "p50_s": round(_percentil(todos_valores, 50), 2),
        "p95_s": round(_percentil(todos_valores, 95), 2),
    }
    return resumo


def rodar_todos(casos: list = None, salvar_json: bool = True) -> list:
    """Roda todos os casos de CASOS (ou uma lista passada) e imprime um
    resumo no final, incluindo latência por etapa e por modo de execução.
    Constrói o grafo uma vez só (barato — só monta o StateGraph, não
    recarrega nenhum modelo).

    Se salvar_json=True (padrão), grava um resumo estruturado em
    resultados/execucoes/<modo>_<timestamp>.json — pensado pra depois
    comparar numericamente uma rodada em CPU_LOCAL com uma em GPU_LOCAL
    (ou OPENAI), sem depender de guardar a saída impressa do Colab."""
    casos = casos if casos is not None else CASOS
    modo = obter_modo_ativo()
    print(f"Modo de execução ativo: {modo.value}")
    print(f"Rodando {len(casos)} caso(s)...")

    agente = construir_grafo()

    t_inicio_lote = time.time()
    relatorios = [rodar_caso(agente, caso) for caso in casos]
    duracao_lote = time.time() - t_inicio_lote

    resumo_latencias = _resumo_latencias(relatorios)

    print(f"\n{'=' * 60}")
    print("RESUMO")
    print("=" * 60)
    com_erro = [r for r in relatorios if r["erro"]]
    print(f"Modo: {modo.value}")
    print(f"{len(relatorios)} casos rodados, {len(com_erro)} com erro de execução.")
    print(f"Tempo total do lote: {duracao_lote:.1f}s ({duracao_lote / 60:.1f}min)")
    if com_erro:
        for r in com_erro:
            print(f"  ❌ {r['id']}: {r['erro']}")

    print("\nLatência por etapa (segundos, agregado de todos os casos):")
    for etapa, stats in resumo_latencias.items():
        if etapa == "_geral":
            continue
        print(f"  {etapa}: n={stats['n']}  média={stats['media_s']}s  "
              f"p50={stats['p50_s']}s  p95={stats['p95_s']}s")
    geral = resumo_latencias["_geral"]
    print(f"  TOTAL (todas as etapas): n={geral['n']}  média={geral['media_s']}s  "
          f"p50={geral['p50_s']}s  p95={geral['p95_s']}s")

    print("\nNota: os campos de texto livre (titulo/objetivo/resultado_esperado/"
          "bloqueios/link_evidencia) precisam de revisão manual — o marcador "
          "🔍/✅ acima é só uma dica, não veredito.")

    if salvar_json:
        os.makedirs(_DIR_RESULTADOS, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        caminho = os.path.join(_DIR_RESULTADOS, f"{modo.value}_{timestamp}.json")
        conteudo = {
            "modo": modo.value,
            "timestamp": timestamp,
            "n_casos": len(relatorios),
            "n_erros": len(com_erro),
            "duracao_lote_s": round(duracao_lote, 2),
            "resumo_latencias": resumo_latencias,
            "casos": relatorios,
        }
        with open(caminho, "w", encoding="utf-8") as f:
            json.dump(conteudo, f, ensure_ascii=False, indent=2)
        print(f"\n💾 resultado salvo em {caminho} — commite esse arquivo pra guardar "
              f"o histórico e poder comparar com outras rodadas (outros modos) depois.")

    return relatorios


if __name__ == "__main__":
    rodar_todos()
