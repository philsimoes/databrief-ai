# ============================================================
# rag/ingestao.py
# DataBrief AI — Bloco 05b: Ingestão do corpus RAG
# Extração + limpeza + chunking do PDF e do site YDUQS,
# geração de embeddings, construção e persistência do índice FAISS.
# Módulo autossuficiente — não depende de variáveis do notebook.
# ============================================================

import io
import json
import os
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import pdfplumber
import requests
from bs4 import BeautifulSoup

# ────────────────────────────────────────────────────────────
# FONTES DO SITE — scraping único, controlado por ATUALIZAR_CORPUS
# ────────────────────────────────────────────────────────────

PAGINAS_SITE = {
    "quem_somos": "https://www.yduqs.com.br/show.aspx?idCanal=U%2FccuSh0iht1%2FmEX%2Fez1ng%3D%3D",
    "nossa_historia": "https://www.yduqs.com.br/show.aspx?idCanal=2A9MtfzS4uRqXhzucRSqdw%3D%3D",
    "estrutura_negocio": "https://www.yduqs.com.br/show.aspx?idCanal=QoTEqrmyYOLej8xm3EvbrA%3D%3D",
}

MARCAS_CONHECIDAS = {"Estácio", "Ibmec", "IDOMED", "Wyden", "Damásio", "Grupo Q", "Hardwork Medicina"}

_HEADERS_REQUEST = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}


# ────────────────────────────────────────────────────────────
# PDF — extração, limpeza, chunking por página
# ────────────────────────────────────────────────────────────

def _limpar_pagina_pdf(texto: str, numero_pagina: int) -> str:
    """Remove número de página solto no final e recola hífen quebrado no fim de linha.

    Validado contra o PDF real da apresentação YDUQS (16/08/2026): a remoção de
    rodapé/cabeçalho repetido por frequência foi testada e descartada — esse PDF
    não tem esse problema (é um deck de slides, não um relatório paginado).
    """
    linhas = texto.split("\n")
    if linhas and linhas[-1].strip() == str(numero_pagina):
        linhas = linhas[:-1]
    texto_limpo = "\n".join(linhas)
    texto_limpo = re.sub(r"(\w)-\n(\w)", r"\1\2", texto_limpo)
    return texto_limpo.strip()


def extrair_chunks_pdf(caminho_pdf: str) -> List[Dict]:
    """Extrai texto do PDF, limpa e faz chunking por página (uma página = um chunk).

    Retorna lista de dicts: {"arquivo", "pagina", "secao", "texto"}.
    `secao` é uma aproximação (primeira linha não vazia da página) — serve como
    rótulo auxiliar de debug, não é usada na busca nem na citação final.
    """
    chunks = []
    with pdfplumber.open(caminho_pdf) as pdf:
        for i, pagina in enumerate(pdf.pages, start=1):
            texto_bruto = pagina.extract_text() or ""
            texto_limpo = _limpar_pagina_pdf(texto_bruto, i)

            linhas = [l.strip() for l in texto_limpo.split("\n") if l.strip()]
            secao_aproximada = linhas[0] if linhas else "(sem título identificado)"

            if len(texto_limpo.strip()) < 10:
                continue  # página vazia/só imagem — não vira chunk

            chunks.append({
                "arquivo": os.path.basename(caminho_pdf),
                "pagina": i,
                "secao": secao_aproximada,
                "texto": texto_limpo,
            })
    return chunks


# ────────────────────────────────────────────────────────────
# SITE — scraping, extração do conteúdo real, chunking por ano/seção
# ────────────────────────────────────────────────────────────

def _extrair_conteudo_principal(html: str) -> Optional[str]:
    """Isola o texto real da página, via o contêiner id="mainContent"
    (comum às 3 páginas do site YDUQS/RI) — descarta menu, aviso de
    resultados trimestrais e widget de "últimas atualizações".
    """
    soup = BeautifulSoup(html, "html.parser")
    container = soup.find(id="mainContent")
    if container is None:
        return None
    texto = container.get_text(separator="\n")
    linhas = [l.strip() for l in texto.split("\n")]
    linhas = [l for l in linhas if l]
    return "\n".join(linhas)


def _chunkar_historia(texto: str) -> List[Dict]:
    """Chunking por ano da página 'Nossa História'.

    A lista de navegação por ano no topo da página tem o mesmo formato do
    ano real do primeiro chunk de conteúdo — usa lookahead (só pula enquanto
    a linha atual E a seguinte forem ambas apenas um ano) para não engolir
    o primeiro ano de conteúdo real junto com a navegação.
    """
    linhas = texto.split("\n")
    padrao_ano = re.compile(r"^(19|20)\d{2}$")

    i = 0
    while i < len(linhas) - 1 and padrao_ano.match(linhas[i].strip()) and padrao_ano.match(linhas[i + 1].strip()):
        i += 1
    linhas = linhas[i:]

    chunks = []
    chunk_atual, ano_atual = [], None
    for linha in linhas:
        if padrao_ano.match(linha.strip()):
            if chunk_atual:
                chunks.append({"secao": ano_atual, "texto": "\n".join(chunk_atual).strip()})
            ano_atual = linha.strip()
            chunk_atual = [linha]
        else:
            chunk_atual.append(linha)
    if chunk_atual:
        chunks.append({"secao": ano_atual, "texto": "\n".join(chunk_atual).strip()})
    return chunks


def _eh_titulo(linha: str) -> bool:
    l = linha.strip()
    if not l or len(l) > 60 or not any(c.isalpha() for c in l):
        return False
    return l == l.upper() or l in MARCAS_CONHECIDAS


def _chunkar_por_titulo(texto: str) -> List[Dict]:
    """Chunking por seção (títulos em maiúsculas + nomes de marca conhecidos,
    que aparecem em Title Case nas visões de marca — ex.: 'Estácio', 'Ibmec').
    Usado nas páginas 'Quem Somos' e 'Estrutura de Negócio'.
    """
    linhas = texto.split("\n")
    chunks = []
    titulo_atual, chunk_atual = "(introdução)", []
    for linha in linhas:
        if _eh_titulo(linha):
            if chunk_atual:
                chunks.append({"secao": titulo_atual, "texto": "\n".join(chunk_atual).strip()})
            titulo_atual, chunk_atual = linha.strip(), []
        else:
            chunk_atual.append(linha)
    if chunk_atual:
        chunks.append({"secao": titulo_atual, "texto": "\n".join(chunk_atual).strip()})
    return chunks


def raspar_chunks_site() -> List[Dict]:
    """Faz scraping das 3 páginas do site YDUQS e retorna os chunks.

    Retorna lista de dicts: {"arquivo", "pagina" (sempre None), "secao", "texto"}.
    Levanta RuntimeError se alguma página não responder com status 200 ou se
    o contêiner de conteúdo não for encontrado — sinal de que o site mudou de
    estrutura e o seletor `id="mainContent"` precisa ser revisado.
    """
    chunks = []

    for chave, url in PAGINAS_SITE.items():
        resposta = requests.get(url, headers=_HEADERS_REQUEST, timeout=20)
        if resposta.status_code != 200:
            raise RuntimeError(f"Scraping de '{chave}' falhou: status {resposta.status_code}")

        conteudo = _extrair_conteudo_principal(resposta.text)
        if conteudo is None:
            raise RuntimeError(
                f"Contêiner 'mainContent' não encontrado em '{chave}' — "
                "o site pode ter mudado de estrutura."
            )

        if chave == "nossa_historia":
            secoes = _chunkar_historia(conteudo)
        else:
            secoes = _chunkar_por_titulo(conteudo)

        for s in secoes:
            if len(s["texto"].strip()) < 10:
                continue
            chunks.append({
                "arquivo": f"site:{chave}",
                "pagina": None,
                "secao": s["secao"],
                "texto": s["texto"],
            })

    return chunks


# ────────────────────────────────────────────────────────────
# CORPUS UNIFICADO
# ────────────────────────────────────────────────────────────

def construir_corpus(caminho_pdf: str) -> List[Dict]:
    """Extrai e unifica os chunks do PDF e do site num corpus único,
    com esquema consistente: {"arquivo", "pagina", "secao", "texto"}.
    """
    chunks_pdf = extrair_chunks_pdf(caminho_pdf)
    chunks_site = raspar_chunks_site()
    corpus = chunks_pdf + chunks_site
    return corpus


# ────────────────────────────────────────────────────────────
# EMBEDDINGS E ÍNDICE FAISS
# ────────────────────────────────────────────────────────────

_MODELO_EMBEDDINGS_NOME = "intfloat/multilingual-e5-small"


def carregar_modelo_embeddings():
    """Carrega o multilingual-e5-small (CPU, ~470MB).

    Nota de ambiente Colab: `sentence-transformers` colide com `torchvision`/
    `Pillow` pré-instalados. Antes da PRIMEIRA importação de sentence_transformers
    na sessão, rode `!pip uninstall -y torchvision -q` numa célula própria —
    desinstalar depois de uma tentativa de import já ter acontecido na mesma
    sessão não resolve (o Python cacheia a checagem de disponibilidade).
    """
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(_MODELO_EMBEDDINGS_NOME, device="cpu")


def gerar_embeddings(corpus: List[Dict], modelo_embeddings=None) -> Tuple[np.ndarray, object]:
    """Gera embeddings normalizados (L2) para todos os chunks do corpus.
    Retorna (embeddings, modelo_embeddings) — o modelo é retornado para reuso
    nas buscas subsequentes, sem precisar recarregar.
    """
    if modelo_embeddings is None:
        modelo_embeddings = carregar_modelo_embeddings()

    textos_com_prefixo = [f"passage: {c['texto']}" for c in corpus]
    embeddings = modelo_embeddings.encode(textos_com_prefixo, convert_to_numpy=True)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings.astype("float32"), modelo_embeddings


def construir_indice_faiss(embeddings: np.ndarray):
    """Constrói um índice FAISS (produto interno = similaridade de cosseno,
    já que os embeddings vêm normalizados)."""
    import faiss
    dimensao = embeddings.shape[1]
    indice = faiss.IndexFlatIP(dimensao)
    indice.add(embeddings)
    return indice


# ────────────────────────────────────────────────────────────
# PERSISTÊNCIA NO DRIVE
# Corpus (JSON) + índice FAISS. Célula de setup do notebook verifica
# se esses arquivos existem antes de decidir se reconstrói.
# ────────────────────────────────────────────────────────────

_NOME_ARQUIVO_CORPUS = "corpus.json"
_NOME_ARQUIVO_INDICE = "indice.faiss"


def salvar_no_drive(corpus: List[Dict], indice, pasta_drive: str) -> None:
    """Salva o corpus (metadados + texto) e o índice FAISS na pasta do Drive."""
    import faiss
    os.makedirs(pasta_drive, exist_ok=True)

    caminho_corpus = os.path.join(pasta_drive, _NOME_ARQUIVO_CORPUS)
    with open(caminho_corpus, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)

    caminho_indice = os.path.join(pasta_drive, _NOME_ARQUIVO_INDICE)
    faiss.write_index(indice, caminho_indice)


def carregar_do_drive(pasta_drive: str) -> Optional[Tuple[List[Dict], object]]:
    """Carrega corpus + índice do Drive, se ambos os arquivos existirem.
    Retorna None se algum estiver ausente (sinal para reconstruir).
    """
    import faiss

    caminho_corpus = os.path.join(pasta_drive, _NOME_ARQUIVO_CORPUS)
    caminho_indice = os.path.join(pasta_drive, _NOME_ARQUIVO_INDICE)

    if not (os.path.exists(caminho_corpus) and os.path.exists(caminho_indice)):
        return None

    with open(caminho_corpus, "r", encoding="utf-8") as f:
        corpus = json.load(f)
    indice = faiss.read_index(caminho_indice)

    return corpus, indice


# ────────────────────────────────────────────────────────────
# PONTO DE ENTRADA PRINCIPAL
# Chamado pela célula de setup do notebook antes de iniciar o Gradio.
# ────────────────────────────────────────────────────────────

def preparar_corpus_e_indice(
    caminho_pdf: str,
    pasta_drive: str,
    forcar_reconstrucao: bool = False,
) -> Tuple[List[Dict], object, object]:
    """
    Prepara o corpus e o índice FAISS, reaproveitando o que já está salvo
    no Drive sempre que possível.

    Retorna: (corpus, indice_faiss, modelo_embeddings)

    `forcar_reconstrucao=True` corresponde à variável ATUALIZAR_CORPUS do
    plano — ignora o que está salvo e refaz a ingestão (scraping + extração
    do PDF + embeddings) do zero.
    """
    modelo_embeddings = carregar_modelo_embeddings()

    if not forcar_reconstrucao:
        resultado_salvo = carregar_do_drive(pasta_drive)
        if resultado_salvo is not None:
            corpus, indice = resultado_salvo
            print(f"✅ Corpus e índice carregados do Drive ({len(corpus)} chunks).")
            return corpus, indice, modelo_embeddings

    print("Corpus/índice ausente ou reconstrução forçada — gerando do zero...")
    corpus = construir_corpus(caminho_pdf)
    embeddings, modelo_embeddings = gerar_embeddings(corpus, modelo_embeddings)
    indice = construir_indice_faiss(embeddings)
    salvar_no_drive(corpus, indice, pasta_drive)
    print(f"✅ Corpus gerado e salvo no Drive ({len(corpus)} chunks).")

    return corpus, indice, modelo_embeddings
