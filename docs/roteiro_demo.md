# DataBrief AI — Roteiro da demo

**Meta:** 9 cenas em até 12 minutos, modo `GPU_LOCAL`.
**Plano B:** obrigatório — ver seção no fim. O risco mais provável é a GPU gratuita não estar disponível na hora.

## Antes da apresentação (no dia, 30 min antes)

1. Abrir o notebook no Colab, ligar a T4 e rodar **Executar tudo**. Se não houver GPU, ir direto para o Plano B, nível 1.
2. Abrir o link do Gradio numa aba e deixar a interface pronta, sem nada digitado.
3. Deixar à mão, na área de trabalho:
   - `docs/demo/anexo_demo.txt` (anexo da cena 5)
   - `docs/demo/anexo_corrompido.pdf` (caso de falha da cena 9)
   - um áudio curto gravado por você (cena 4) — ver texto sugerido abaixo
   - o relatório `resultados/avaliacoes/GPU_LOCAL_20260907_184439_final.json` aberto no GitHub
4. Fazer uma conversa de aquecimento completa e clicar em **Novo caso (limpa tudo)**. A primeira chamada ao modelo é a mais lenta; o aquecimento tira esse atraso da frente da banca.

**Texto sugerido para o áudio (cena 4), uns 10 segundos:**
> "É pra diretoria comercial, quero ver por polo e por modalidade, EaD e semipresencial."

## Cenas

| # | Cena | O que fazer | O que dizer | Tempo |
|---|---|---|---|---|
| 1 | **Problema** | Mostrar um pedido vago (o texto da cena 3) ao lado de um briefing pronto (JSON de um ensaio) | "A demanda chega assim. Precisa sair assim. Hoje quem faz essa ponte é uma reunião." | 1 min |
| 2 | **Setup** | Mostrar o notebook já executado: modo ativo, GPU detectada. Apontar o aviso de link público no topo da interface | "Roda no Colab gratuito. O modo é escolhido uma vez e nunca troca sozinho." | 1 min |
| 3 | **Texto** | Digitar: *"Preciso acompanhar a evolução da taxa de evasão do Digital mês a mês."* Mostrar o agente perguntando só o que falta e os botões de tipo de demanda | "Ele não pergunta o que já sabe. Campos com opções fixas viram botões — o modelo errava esses." | 2 min |
| 4 | **Áudio** | Enviar o áudio gravado. Mostrar a tela de revisão da transcrição e corrigir uma palavra antes de enviar | "O Whisper carrega, transcreve e libera a GPU. Nada entra na conversa sem revisão." | 2 min |
| 5 | **Anexo** | Clicar em **Novo caso**, enviar `anexo_demo.txt`, mostrar o texto extraído e enviar | "Anexo passa pela mesma revisão. Cada campo vai lembrar que veio do arquivo." | 1 min |
| 6 | **Proveniência** | Responder até o briefing aparecer. Percorrer os badges: *anexo*, *selecionado*, *inferido pelo sistema* | "Cada campo diz de onde veio. É isso que o humano revisa antes de aprovar." | 1 min |
| 7 | **Aprovação** | Clicar em **Editar título / objetivo**, deixar o título mais específico (ex.: incluir "São Paulo"), aprovar. Mostrar o badge *selecionado* no título e baixar o JSON | "O título gerado pelo modelo local tende a ser vago — foi o que puxou a concordância humana para 0,70. Por isso é editável." | 1 min |
| 8 | **Áudio final** | Clicar em **Ouvir resumo**, tocar alguns segundos, apontar o rótulo de voz sintética | "O TTS só existe depois da aprovação, e é rotulado como IA." | 1 min |
| 9 | **Avaliação e falha** | Mostrar a tabela de métricas do README. Depois enviar `anexo_corrompido.pdf` e mostrar a mensagem de erro amigável | "10 casos, dois modos avaliados por completo. E quando dá errado, avisa — não quebra calado." | 2 min |

**Total: 12 min.** Se o tempo apertar, a cena 8 é a primeira a sair.

## Plano B

Gatilhos: Colab sem GPU, link do Gradio cai, modelo trava ou demora demais.

**Nível 1 — trocar de modo (perde ~3 min).** Se houver chave da OpenAI: `MODO_EXECUCAO = "OPENAI"` na Célula 1 e **Executar tudo**. A interface é a mesma e o modo aparece no topo. Dizer em voz alta que trocou e por quê — isso é a regra "nunca fallback silencioso" funcionando ao vivo.

**Nível 2 — sem modelo nenhum.** Narrar o fluxo com o material gravado no ensaio:
- capturas de tela de cada cena (fazer no ensaio, uma por cena)
- o JSON de briefing baixado no ensaio
- o arquivo de áudio do resumo gerado no ensaio
- a transcrição revisada (captura da tela de revisão)

**Nível 3 — sem internet.** Apresentar só com as capturas e o README impresso ou salvo em PDF.

## Checklist de ensaio

Fazer pelo menos 2 ensaios completos, cronometrados.

- [ ] Ensaio 1 completo, tempo anotado: ____ min
- [ ] Capturas de tela das 9 cenas salvas numa pasta
- [ ] JSON de briefing e áudio do resumo do ensaio salvos
- [ ] Ensaio 2 completo, tempo anotado: ____ min
- [ ] Plano B nível 1 testado ao menos uma vez (trocar de modo e rodar tudo)
- [ ] `pytest` rodado e passando no mesmo runtime
