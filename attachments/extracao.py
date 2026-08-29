# ============================================================
# attachments/extracao.py
# DataBrief AI — Bloco 06: Extração de texto de anexos
# Extrai texto de PDF, DOCX ou TXT para entrar no mesmo pipeline
# de extração de campos usado por texto digitado e áudio transcrito.
# Módulo autossuficiente — não depende de variáveis do notebook.
# ============================================================

import os
import time

# Cap de segurança — evita sobrecarregar o prompt de extração do Qwen com um
# documento inteiro. O usuário revisa e pode editar/reduzir o texto antes de
# enviar de qualquer forma, então isso só protege contra o caso extremo.
LIMITE_CARACTERES = 6000

EXTENSOES_SUPORTADAS = (".pdf", ".docx", ".txt")


def _extrair_pdf(caminho: str) -> str:
    import pdfplumber
    partes = []
    with pdfplumber.open(caminho) as pdf:
        for pagina in pdf.pages:
            partes.append(pagina.extract_text() or "")
    return "\n".join(partes).strip()


def _extrair_docx(caminho: str) -> str:
    import docx
    documento = docx.Document(caminho)
    return "\n".join(p.text for p in documento.paragraphs).strip()


def _extrair_txt(caminho: str) -> str:
    with open(caminho, "r", encoding="utf-8", errors="ignore") as f:
        return f.read().strip()


def extrair_texto_anexo(caminho_arquivo: str) -> dict:
    """
    Extrai o texto de um anexo (PDF, DOCX ou TXT) para entrar no pipeline de
    extração de campos, com origem == ATTACHMENT no painel de proveniência.

    Trunca em LIMITE_CARACTERES para não sobrecarregar o prompt de extração
    do Qwen com um documento inteiro.

    Args:
        caminho_arquivo: caminho do arquivo anexado (.pdf, .docx ou .txt)

    Returns:
        dict com:
          "texto": str — texto extraído (truncado se necessário)
          "arquivo": str — nome do arquivo (sem caminho)
          "truncado": bool — True se o texto foi cortado por exceder o limite
          "latencia_s": float

    Levanta ValueError se a extensão não for suportada.
    """
    t0 = time.time()
    nome = os.path.basename(caminho_arquivo)
    extensao = os.path.splitext(caminho_arquivo)[1].lower()

    if extensao == ".pdf":
        texto = _extrair_pdf(caminho_arquivo)
    elif extensao == ".docx":
        texto = _extrair_docx(caminho_arquivo)
    elif extensao == ".txt":
        texto = _extrair_txt(caminho_arquivo)
    else:
        raise ValueError(
            f"Tipo de arquivo não suportado: '{extensao}'. "
            f"Use um dos formatos: {', '.join(EXTENSOES_SUPORTADAS)}."
        )

    truncado = len(texto) > LIMITE_CARACTERES
    if truncado:
        texto = texto[:LIMITE_CARACTERES]

    latencia = time.time() - t0
    return {
        "texto": texto,
        "arquivo": nome,
        "truncado": truncado,
        "latencia_s": round(latencia, 2),
    }
