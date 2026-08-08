# DataBrief AI

Agente conversacional multimodal para refinamento de demandas de dados em briefings estruturados.

**Desenvolvedor:** Phil — Gerência de Inteligência de Dados, YDUQS  
**Runtime:** Google Colab gratuito (GPU T4)  
**Status:** 🚧 Em desenvolvimento — Ato 1

---

## Estrutura do repositório

```
databrief-ai/
├── schemas/          # Pydantic: DemandState, FieldProvenance, TurnInput, BriefingOutput
├── graph/            # Grafo LangGraph — nós e arestas do agente
├── rag/              # Pipeline RAG: embeddings E5-small + FAISS
├── audio/            # Whisper ASR + Piper TTS
├── interface/        # Gradio Blocks
├── evaluation/       # Dataset de avaliação, gabarito, métricas
├── corpus/
│   └── aurora_varejo/  # Corpus fictício para desenvolvimento (Atos 1–4)
├── briefings/        # Briefings gerados (não versionados — ficam no Drive)
├── logs/             # Logs de sessão (não versionados)
├── scripts/          # Scripts auxiliares (setup, avaliação)
├── requirements.txt
└── README.md
```

## Modos de execução

| Modo | LLM | ASR | GPU necessária |
|------|-----|-----|----------------|
| `GPU_LOCAL` | Qwen3-4B 4-bit | Whisper Small | T4 Colab |
| `CPU_LOCAL` | Qwen3-1.7B | Whisper Tiny | Não |
| `OPENAI` | gpt-4o-mini | Whisper local | Não |

## Como executar

Documentação completa disponível no Ato 4.
