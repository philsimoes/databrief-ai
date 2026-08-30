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
from schemas.models import ModoExecucao
from graph.agent import obter_modo_ativo

# Tamanho do Whisper por modo de execução — GPU_LOCAL usa Small (mais preciso,
# GPU tem folga de VRAM); CPU_LOCAL usa Tiny (bem mais leve/rápido, essencial
# rodando só em CPU); OPENAI usa Small (mesmo critério do GPU_LOCAL — a API
# cobre só a geração de texto, a transcrição continua local).
_MODELOS_WHISPER = {
    ModoExecucao.GPU_LOCAL: "openai/whisper-small",
    ModoExecucao.CPU_LOCAL: "openai/whisper-tiny",
    ModoExecucao.OPENAI:    "openai/whisper-small",
}


def _vram():
    if not torch.cuda.is_available():
        return {"alocada_gb": 0.0, "livre_gb": 0.0}
    return {
        "alocada_gb": round(torch.cuda.memory_allocated() / 1024**3, 2),
        "livre_gb": round(torch.cuda.mem_get_info()[0] / 1024**3, 2),
    }


def transcrever_audio(caminho_audio, idioma="portuguese", verbose=True, modelo_whisper=None):
    """
    Carrega o Whisper (Small ou Tiny, conforme o modo de execução ativo),
    transcreve o arquivo de áudio e libera a GPU antes de retornar.

    Args:
        caminho_audio: caminho do arquivo de áudio (mp3, m4a, wav etc.)
        idioma: idioma da transcrição
        verbose: se True, imprime VRAM e latência no console
        modelo_whisper: força um modelo específico (ex.: para testes) — por
            padrão usa o mapeamento por modo (_MODELOS_WHISPER), consultando
            o modo realmente ativo via obter_modo_ativo() (nunca adivinhado)

    Returns:
        dict com "texto" (str) e "latencia_s" (float)
    """
    import librosa

    modelo_whisper = modelo_whisper or _MODELOS_WHISPER[obter_modo_ativo()]

    t0 = time.time()
    tipo_dado = torch.float16 if torch.cuda.is_available() else torch.float32

    processor = AutoProcessor.from_pretrained(modelo_whisper)
    modelo = AutoModelForSpeechSeq2Seq.from_pretrained(
        modelo_whisper, torch_dtype=tipo_dado, low_cpu_mem_usage=True, use_safetensors=True
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    transcrevedor = hf_pipeline(
        "automatic-speech-recognition", model=modelo, tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor, torch_dtype=tipo_dado,
        chunk_length_s=30, batch_size=8,
    )
    if verbose:
        v = _vram()
        print(f"📊 [Whisper carregado: {modelo_whisper}] {v['alocada_gb']} GB alocados | {v['livre_gb']} GB livres")

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
