# ============================================================
# testes/conftest.py
# DataBrief AI — Bloco 32: configuração compartilhada do pytest
#
# Roda sem GPU e sem modelo: o LLM nunca é chamado nestes testes. Os módulos
# de áudio (Whisper/Piper) são substituídos por versões falsas só durante os
# testes, porque importam torch/transformers e baixam modelos.
#
# Uso no Colab (depois da Célula 1):
#   !cd /content/databrief-ai && python -m pytest testes -q
# ============================================================

import os
import sys
import types

import pytest

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if RAIZ not in sys.path:
    sys.path.insert(0, RAIZ)


def _stub(nome, funcao):
    if nome not in sys.modules:
        modulo = types.ModuleType(nome)
        setattr(modulo, funcao, lambda *a, **k: None)
        sys.modules[nome] = modulo


@pytest.fixture(scope="session", autouse=True)
def modo_registrado():
    """Registra GPU_LOCAL com objetos falsos — suficiente pra obter_modo_ativo()."""
    import graph.agent as ag
    from schemas.models import ModoExecucao
    ag.inicializar_modelo(object(), object(), ModoExecucao.GPU_LOCAL)
    yield


@pytest.fixture()
def app():
    """interface.app com áudio substituído por funções vazias."""
    _stub("audio.transcricao", "transcrever_audio")
    _stub("audio.sintese", "sintetizar_texto")
    import interface.app as modulo
    return modulo
