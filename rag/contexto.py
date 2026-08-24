# ============================================================
# rag/contexto.py
# DataBrief AI — Bloco 05b: Busca de contexto RAG + julgamento de relevância
# Busca no índice FAISS e usa o Qwen (já carregado pelo grafo) para decidir
# quais trechos são relevantes e gerar a citação legível.
# Módulo autossuficiente — não depende de variáveis do notebook.
# ============================================================

from typing import Callable, Dict, List, Tuple

import numpy as np


# ────────────────────────────────────────────────────────────
# PROMPT DE JULGAMENTO
# Versão v3 — validada contra os 9 casos-âncora (16/08/2026):
# texto completo dos candidatos (sem corte), instrução explícita contra
# inventar associação número/ano em texto bagunçado de página com gráfico,
# e pedido de citar cada fonte uma única vez.
# ────────────────────────────────────────────────────────────

PROMPT_JULGAMENTO = """Você recebe a demanda de dados de um usuário e trechos candidatos de um corpus de contexto de mercado.

Um trecho só é relevante se contiver um DADO OU INFORMAÇÃO CONCRETA relacionada à demanda (número, fato específico, evento, nome). Não conte como relevante um trecho que só menciona um tema parecido de forma genérica (ex.: valores institucionais, missão, visão), mesmo que use palavras parecidas.

Atenção: alguns trechos vêm de páginas com gráficos, e o texto extraído pode estar desorganizado, com números soltos sem estarem claramente ligados ao rótulo/ano/métrica certos. Se não tiver certeza de qual número corresponde a qual ano ou métrica, NÃO invente essa associação — descreva o tema geral do trecho sem cravar números específicos. É melhor uma citação genérica e correta do que uma específica e possivelmente errada.

Para cada trecho relevante, escreva uma linha: "Fonte: [nome da fonte] — [citação curta de 1-2 frases]". Cite cada fonte no máximo uma vez.
Se nenhum trecho tiver informação relevante e confiável, responda apenas: "Nenhum trecho relevante."
Não invente informação que não esteja no texto.

Demanda do usuário: {demanda}

{blocos}

Resposta:"""


# ────────────────────────────────────────────────────────────
# BUSCA POR EMBEDDING
# ────────────────────────────────────────────────────────────

def buscar_candidatos(
    objetivo: str,
    corpus: List[Dict],
    indice,
    modelo_embeddings,
    k: int = 5,
) -> List[Dict]:
    """Busca os k chunks mais próximos do objetivo da demanda no índice FAISS.

    Usa o prefixo "query: " exigido pelo modelo e5. Retorna os chunks do
    corpus correspondentes aos índices encontrados (ordem = mais relevante
    primeiro, segundo a similaridade de cosseno).
    """
    query_emb = modelo_embeddings.encode([f"query: {objetivo}"], convert_to_numpy=True)
    query_emb = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True)
    query_emb = query_emb.astype("float32")

    k_efetivo = min(k, len(corpus))
    _, indices = indice.search(query_emb, k_efetivo)

    return [corpus[idx] for idx in indices[0] if idx != -1]


def montar_prompt_julgamento(objetivo: str, candidatos: List[Dict]) -> str:
    """Monta o prompt de julgamento de relevância a partir dos candidatos —
    texto completo de cada chunk, sem corte (corte em 500 caracteres foi
    testado e causou falso-negativo em conteúdo genuinamente relevante).
    """
    blocos = []
    for i, c in enumerate(candidatos, start=1):
        fonte = c["arquivo"]
        fonte += f" (pág. {c['pagina']})" if c.get("pagina") else f" ({c['secao']})"
        blocos.append(f"Trecho {i} — Fonte: {fonte}\n{c['texto']}")

    return PROMPT_JULGAMENTO.format(demanda=objetivo, blocos="\n".join(blocos))


# ────────────────────────────────────────────────────────────
# PARSER DA RESPOSTA DO QWEN
# ────────────────────────────────────────────────────────────

def parse_resposta_citacoes(texto_resposta: str) -> List[Dict]:
    """Extrai as citações da resposta do Qwen no formato
    "Fonte: [nome] — [citação]" por linha. Retorna lista vazia se o Qwen
    respondeu "Nenhum trecho relevante." (comportamento de abstenção,
    validado nos casos-âncora sem contexto real no corpus).
    """
    texto = (texto_resposta or "").strip()
    if not texto or "nenhum trecho relevante" in texto.lower():
        return []

    citacoes = []
    for linha in texto.split("\n"):
        linha = linha.strip()
        if not linha.lower().startswith("fonte:"):
            continue
        conteudo = linha[len("fonte:"):].strip()

        fonte, citacao = conteudo, ""
        for separador in (" — ", " – ", " - "):
            if separador in conteudo:
                fonte, citacao = conteudo.split(separador, 1)
                break

        citacoes.append({"fonte": fonte.strip(), "citacao": citacao.strip()})

    return citacoes


# ────────────────────────────────────────────────────────────
# PONTO DE ENTRADA PRINCIPAL
# Chamado pelo nó novo do grafo (graph/agent.py), uma vez, quando a
# demanda fica pronta — não a cada turno.
# ────────────────────────────────────────────────────────────

def buscar_contexto(
    objetivo: str,
    corpus: List[Dict],
    indice,
    modelo_embeddings,
    chamar_llm: Callable[[str, int], Tuple[str, float]],
    k: int = 5,
    max_tokens: int = 350,
) -> List[Dict]:
    """
    Busca contexto relevante no corpus para o objetivo da demanda e retorna
    as citações que o LLM julgou relevantes (lista vazia se nada relevante
    ou confiável foi encontrado — comportamento de abstenção é intencional).

    `chamar_llm` é uma função (prompt, max_tokens) -> (resposta, latência) —
    passar `graph.agent.chamar_qwen` para reaproveitar o Qwen já carregado,
    sem precisar de uma segunda cópia do modelo na GPU.

    Retorna lista de dicts: {"fonte": str, "citacao": str}.
    """
    if not objetivo or not objetivo.strip():
        return []

    candidatos = buscar_candidatos(objetivo, corpus, indice, modelo_embeddings, k=k)
    if not candidatos:
        return []

    prompt = montar_prompt_julgamento(objetivo, candidatos)
    resposta, _ = chamar_llm(prompt, max_tokens)

    return parse_resposta_citacoes(resposta)
