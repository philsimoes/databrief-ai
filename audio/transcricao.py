# ============================================================
# audio/transcricao.py
# DataBrief AI — Ato 2, Bloco 03: Transcrição de áudio
# Carrega o Whisper sob demanda, transcreve, libera a GPU.
# O Qwen não é gerenciado aqui — continua carregado desde a
# Célula 2 do notebook (decisão registrada: carga sequencial
# estrita não é mais obrigatória, VRAM tem folga de sobra).
# ============================================================

import gc
import time
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline as hf_pipeline

MODELO_WHISPER = "openai/whisper-small"


def _vram():
    if not torch.cuda.is_available():
        return {"alocada_gb": 0.0, "livre_gb": 0.0}
    return {
        "alocada_gb": round(torch.cuda.memory_allocated() / 1024**3, 2),
        "livre_gb": round(torch.cuda.mem_get_info()[0] / 1024**3, 2),
    }


def transcrever_audio(caminho_audio, idioma="portuguese", verbose=True):
    """
    Carrega o Whisper Small, transcreve o arquivo de áudio e libera a GPU
    antes de retornar.

    Args:
        caminho_audio: caminho do arquivo de áudio (mp3, m4a, wav etc.)
        idioma: idioma da transcrição
        verbose: se True, imprime VRAM e latência no console

    Returns:
        dict com "texto" (str) e "latencia_s" (float)
    """
    import librosa

    t0 = time.time()
    tipo_dado = torch.float16 if torch.cuda.is_available() else torch.float32

    processor = AutoProcessor.from_pretrained(MODELO_WHISPER)
    modelo = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODELO_WHISPER, torch_dtype=tipo_dado, low_cpu_mem_usage=True, use_safetensors=True
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    transcrevedor = hf_pipeline(
        "automatic-speech-recognition", model=modelo, tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor, torch_dtype=tipo_dado,
        chunk_length_s=30, batch_size=8,
    )
    if verbose:
        v = _vram()
        print(f"📊 [Whisper carregado] {v['alocada_gb']} GB alocados | {v['livre_gb']} GB livres")

    audio, sr = librosa.load(caminho_audio, sr=16000)
    resultado = transcrevedor(audio, generate_kwargs={"language": idioma})
    texto = resultado["text"].strip()

    del transcrevedor, modelo, processor
    gc.collect()
    torch.cuda.empty_cache()

    latencia = time.time() - t0
    if verbose:
        v = _vram()
        print(f"📊 [Whisper liberado] {v['alocada_gb']} GB alocados | {v['livre_gb']} GB livres | {latencia:.1f}s total")

    return {"texto": texto, "latencia_s": round(latencia, 2)}
