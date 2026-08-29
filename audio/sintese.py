# ============================================================
# audio/sintese.py
# DataBrief AI — Bloco 06: Síntese de voz do resumo do briefing (Piper TTS)
# Roda em CPU (não disputa VRAM com o Qwen, que fica carregado o tempo todo).
# Carrega a voz só quando chamado (depois da aprovação do briefing) e libera
# logo em seguida — mesmo princípio de carga sob demanda já usado no Whisper
# (audio/transcricao.py).
# Módulo autossuficiente — não depende de variáveis do notebook.
# ============================================================

import os
import time
import wave

# Voz padrão: pt_BR-faber-medium — única qualidade disponível para essa voz
# no catálogo oficial do Piper (rhasspy/piper-voices), voz masculina única.
VOZ_PADRAO = "pt_BR-faber-medium"

# Diretório onde o modelo de voz fica salvo entre sessões — dentro do Drive
# montado pelo notebook, mesmo padrão usado pro corpus/índice (quando havia
# RAG). Se o Drive não estiver montado (ex.: rodando fora do Colab), cai para
# uma pasta local — nesse caso a voz é baixada de novo a cada sessão nova.
DIRETORIO_VOZ_PADRAO = (
    "/content/drive/MyDrive/databrief-ai/piper_voices"
    if os.path.isdir("/content/drive/MyDrive")
    else "/tmp/databrief_piper_voices"
)


def _garantir_voz_baixada(nome_voz: str, diretorio: str) -> tuple:
    """
    Baixa o modelo de voz (.onnx + .onnx.json) do catálogo oficial do Piper
    se ainda não existir no diretório. Idempotente — se os arquivos já
    existem, não baixa de novo (a própria função download_voice do piper-tts
    já faz essa checagem, mas confirmamos aqui também para não depender só
    do comportamento interno da lib).

    Retorna (caminho_modelo, caminho_config).
    """
    from piper.download_voices import download_voice

    os.makedirs(diretorio, exist_ok=True)
    caminho_modelo = os.path.join(diretorio, f"{nome_voz}.onnx")
    caminho_config = os.path.join(diretorio, f"{nome_voz}.onnx.json")

    if not (os.path.exists(caminho_modelo) and os.path.exists(caminho_config)):
        download_voice(nome_voz, diretorio)

    return caminho_modelo, caminho_config


def sintetizar_texto(
    texto: str,
    caminho_saida_wav: str,
    nome_voz: str = VOZ_PADRAO,
    diretorio_voz: str = DIRETORIO_VOZ_PADRAO,
) -> dict:
    """
    Sintetiza texto em áudio (WAV) usando Piper TTS.

    Carrega a voz (CPU), gera o áudio, e libera a voz da memória antes de
    retornar — não mantém o modelo residente entre chamadas. Pensado para
    ser chamado uma vez por briefing aprovado (não em todo turno).

    Args:
        texto: texto a sintetizar (o resumo executivo do briefing aprovado)
        caminho_saida_wav: caminho onde salvar o arquivo .wav gerado
        nome_voz: nome da voz Piper, formato "idioma-nome-qualidade"
        diretorio_voz: pasta onde o modelo de voz é baixado/cacheado

    Returns:
        dict com:
          "caminho_audio": str — mesmo valor de caminho_saida_wav
          "latencia_s": float

    Levanta ValueError se o texto estiver vazio.
    """
    if not texto or not texto.strip():
        raise ValueError("Texto vazio — nada para sintetizar.")

    from piper import PiperVoice

    t0 = time.time()
    caminho_modelo, caminho_config = _garantir_voz_baixada(nome_voz, diretorio_voz)

    voz = PiperVoice.load(caminho_modelo, config_path=caminho_config, use_cuda=False)
    try:
        with wave.open(caminho_saida_wav, "wb") as wav_file:
            voz.synthesize_wav(texto, wav_file)
    finally:
        # Libera a voz da memória logo após o uso — não fica residente
        del voz

    latencia = time.time() - t0
    return {
        "caminho_audio": caminho_saida_wav,
        "latencia_s": round(latencia, 2),
    }
