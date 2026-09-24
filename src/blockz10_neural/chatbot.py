"""
Blockz10 Chat — retrieval chatbot routed by the block network.

Architecture (honest by design):
  * The ANSWERS are pre-written (bilingual, about the Blockz10
    ecosystem) — a retrieval bot, not a generative model.
  * The ROUTING is neural and runs on the pyramid: the question is
    reduced to 10 keyword-family scores (features), the BlockNet
    redistributes the deposit for 3 rounds, and the intent is the base
    block where the value lands.
  * Zero keyword hits, or low confidence => fallback answer. The
    network never invents content.

Author : Joaquim Pedro de Morais Filho <j360074@hotmail.com>
License: MIT
"""

from __future__ import annotations

import unicodedata

import numpy as np

INTENTS = ["conceito", "codificacao", "lottery", "piramide", "neural"]

# 10 keyword families = the 10 input blocks of the pyramid.
# Keywords in WORD_ONLY are matched as whole words (to avoid hits
# inside unrelated words, e.g. "use" in "because"); everything else is
# a substring match. All matching is lowercase, accent-free.
WORD_ONLY = frozenset({"use", "run", "lab", "rede", "site", "code", "ia", "ai"})
FAMILIES: list[list[str]] = [
    # f0 concept / system
    ["blockz", "sistema", "system", "conceito", "concept", "o que e",
     "what is", "bloco", "block", "funcao", "function", "ideia", "idea",
     "origem", "origin", "manifesto"],
    # f1 encoding {e,1}
    ["codific", "encoding", "encode", "decode", "compress", "eee", "311",
     "alfabeto", "alphabet", "e,1", "e1", "hexadecimal", "hex", "mnemonic"],
    # f2 lottery
    ["loteria", "lottery", "premio", "prize", "anagrama", "anagram",
     "senha", "password", "embaralh", "shuffle", "yourtoken", "carteira",
     "wallet", "tesouro", "treasure", "puzzle", "sorteio"],
    # f3 pyramid / 15/5 / contract
    ["piramide", "pyramid", "15/5", "15 5", "nivel", "level", "distribu",
     "split", "contrato", "contract", "solidity", "conserv", "150",
     "royalt", "pagamento", "payment"],
    # f4 neural / learning
    ["neural", "rede", "network", "trein", "train", "aprend", "learn",
     "xor", "iris", "gradiente", "gradient", "backprop", "classific",
     "inteligencia", "intelligence", "laboratorio", "lab", "modelo",
     "model", "limiar", "threshold", "bonus", "relu", "parametro",
     "parameter", "acuracia", "accuracy"],
    # f5 author
    ["autor", "author", "joaquim", "criador", "creator", "quem criou",
     "quem fez", "who made", "who created", "contato", "contact", "email"],
    # f6 on-chain registry
    ["nft", "opensea", "registro", "register", "on-chain", "onchain",
     "blockchain", "ethereum", "erc-20", "erc20", "token"],
    # f7 usage / code
    ["como us", "how to", "how do", "instal", "github", "python",
     "javascript", "codigo", "code", "repo", "download", "rodar",
     "executar", "usar", "use"],
    # f8 media
    ["video", "site", "demo", "assistir", "watch", "pagina", "page",
     "mostrar", "show"],
    # f9 security / difficulty
    ["seguranca", "security", "entropia", "entropy", "bits",
     "forca bruta", "brute", "seguro", "safe", "risco", "risk",
     "quebrar", "crack", "dificuldade", "difficulty", "dificil", "hard"],
]

CONFIDENCE_FLOOR = 0.50


def normalize(text: str) -> str:
    """Lowercase and strip accents — mirrors the JS implementation."""
    t = unicodedata.normalize("NFD", text.lower())
    return "".join(c for c in t if unicodedata.category(c) != "Mn")


def featurize(question: str) -> tuple[np.ndarray, int]:
    """10 keyword-family scores in [0,1] plus the total hit count."""
    t = normalize(question)
    words = set(t.replace("?", " ").replace("!", " ").replace(",", " ")
                 .replace(".", " ").split())
    hits = np.zeros(len(FAMILIES))
    for i, fam in enumerate(FAMILIES):
        for kw in fam:
            if kw in WORD_ONLY:
                if kw in words:
                    hits[i] += 1
            elif kw in t:
                hits[i] += 1
    total = int(hits.sum())
    return hits / max(1, total), total


# ---------------------------------------------------------------------------
# Synthetic training questions (PT + EN), ~30 per intent.
# ---------------------------------------------------------------------------

QUESTIONS: list[tuple[str, int]] = [
    # --- 0 · conceito -------------------------------------------------------
    ("o que é o blockz10?", 0), ("o que é esse sistema de blocos?", 0),
    ("qual é o conceito do projeto?", 0), ("explica a ideia do blockz10", 0),
    ("o que é um bloco nesse sistema?", 0), ("qual a função dos blocos?", 0),
    ("de onde surgiu o blockz10?", 0), ("qual a origem do sistema?", 0),
    ("quem é o autor do blockz10?", 0), ("quem criou esse projeto?", 0),
    ("qual o contato do criador?", 0), ("qual o email do autor?", 0),
    ("o blockz10 está registrado como nft?", 0),
    ("onde está o registro on-chain do conceito?", 0),
    ("o conceito tem nft na opensea?", 0),
    ("me fala sobre o sistema e sua origem", 0),
    ("what is blockz10?", 0), ("what is this block system?", 0),
    ("explain the concept of the project", 0),
    ("what is a block in this system?", 0),
    ("who is the author of blockz10?", 0), ("who created this system?", 0),
    ("how do i contact the creator?", 0),
    ("is the concept registered as an nft?", 0),
    ("where is the on-chain register?", 0),
    ("tell me the idea behind the blocks", 0),
    ("qual é a filosofia do sistema de blocos?", 0),
    ("o que significa block system for creating other functions?", 0),
    ("quem é joaquim pedro de morais filho?", 0),
    ("what function do the blocks create?", 0),
    # --- 1 · codificação ----------------------------------------------------
    ("como funciona a codificação e1?", 1),
    ("o que significa eee11 virar 311?", 1),
    ("explica a compressão do alfabeto e,1", 1),
    ("como codificar uma string nesse alfabeto?", 1),
    ("a codificação é sem perda?", 1),
    ("por que e e 1 são dígitos hexadecimais?", 1),
    ("como decodificar 311?", 1),
    ("qual a regra de compressão dos runs?", 1),
    ("uma chave e1 é uma chave ethereum válida?", 1),
    ("quantos bits de entropia tem uma chave e1?", 1),
    ("a chave do alfabeto restrito é segura?", 1),
    ("posso guardar fundos numa chave e1?", 1),
    ("how does the e1 encoding work?", 1),
    ("why does eee11 become 311?", 1),
    ("is the encoding lossless?", 1),
    ("how do i encode a string in this alphabet?", 1),
    ("is an e1 key a valid ethereum key?", 1),
    ("how many bits of entropy does the restricted key have?", 1),
    ("is the e1 alphabet key safe for real funds?", 1),
    ("what is the compression rule for runs?", 1),
    ("como funciona o encoder do site?", 1),
    ("qual o tamanho da chave depois de comprimida?", 1),
    ("explain the hexadecimal trick of the alphabet", 1),
    ("como gerar uma chave aleatória e1?", 1),
    # --- 2 · lottery --------------------------------------------------------
    ("como funciona a lottery yourtoken?", 2),
    ("o que é a loteria do token?", 2),
    ("como ganho o prêmio da loteria?", 2),
    ("o que é o anagrama de 30 caracteres?", 2),
    ("por que a senha é publicada embaralhada?", 2),
    ("como deposito tokens no prêmio?", 2),
    ("qualquer um pode aumentar o prêmio?", 2),
    ("como verifico um palpite da loteria?", 2),
    ("o que acontece quando alguém acerta a ordem?", 2),
    ("a loteria precisa de servidor?", 2),
    ("quantas ordens possíveis tem o anagrama?", 2),
    ("qual a dificuldade da loteria?", 2),
    ("é difícil quebrar a senha embaralhada por força bruta?", 2),
    ("how does lottery yourtoken work?", 2),
    ("how do i win the lottery prize?", 2),
    ("what is the 30 character anagram?", 2),
    ("why is the password published shuffled?", 2),
    ("can anyone grow the prize wallet?", 2),
    ("how do i verify a lottery guess?", 2),
    ("how hard is it to brute force the shuffled password?", 2),
    ("what happens when someone finds the order?", 2),
    ("me fala da caça ao tesouro criptográfica", 2),
    ("como funciona o sorteio com a carteira prêmio?", 2),
    ("what is the treasure hunt with the prize wallet?", 2),
    # --- 3 · pirâmide -------------------------------------------------------
    ("como funciona a pirâmide 15/5?", 3),
    ("o que é o block 15/5?", 3),
    ("como o valor é distribuído entre os níveis?", 3),
    ("por que a soma sempre fecha em 150?", 3),
    ("o que é a poeira de arredondamento?", 3),
    ("o que acontece no nível 0 da pirâmide?", 3),
    ("por que é uma anti-pirâmide?", 3),
    ("existe contrato solidity da pirâmide?", 3),
    ("como o contrato faz o split de pagamentos?", 3),
    ("o split conserva o valor?", 3),
    ("dá para usar para royalties?", 3),
    ("como funciona a redistribuição conservativa?", 3),
    ("how does the 15/5 pyramid work?", 3),
    ("what is block 15/5?", 3),
    ("how is value distributed across levels?", 3),
    ("why does the sum always return to 150?", 3),
    ("what is the rounding dust?", 3),
    ("is there a solidity contract for the pyramid?", 3),
    ("can i use it for payment splits?", 3),
    ("does the split conserve value?", 3),
    ("why is it an anti-pyramid?", 3),
    ("qual a diferença para uma pirâmide financeira?", 3),
    ("how does the conservative redistribution work?", 3),
    ("quais são as regras de divisão dos níveis?", 3),
    # --- 4 · neural ---------------------------------------------------------
    ("como funciona a rede neural de blocos?", 4),
    ("o que é o blockz10 neural?", 4),
    ("a pirâmide consegue aprender?", 4),
    ("como a rede foi treinada?", 4),
    ("o que é o bônus por limiar?", 4),
    ("qual a acurácia no íris?", 4),
    ("a rede resolve o xor?", 4),
    ("como funciona o backprop manual?", 4),
    ("posso treinar a rede no navegador?", 4),
    ("onde fica o laboratório de treino?", 4),
    ("quantos parâmetros tem o modelo?", 4),
    ("a rede é uma inteligência artificial?", 4),
    ("como o gradiente foi verificado?", 4),
    ("how does the block neural network work?", 4),
    ("what is blockz10 neural?", 4),
    ("can the pyramid learn?", 4),
    ("how was the network trained?", 4),
    ("what is the threshold bonus?", 4),
    ("what accuracy does it get on iris?", 4),
    ("does the network solve xor?", 4),
    ("can i train it in the browser?", 4),
    ("how many parameters does the model have?", 4),
    ("how was the gradient verified?", 4),
    ("what is the learning rule of the blocks?", 4),
    ("a rede classifica padrões e1?", 4),
]


def dataset(seed: int = 155, test_frac: float = 0.2):
    """Featurized train/test split of the synthetic questions."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(QUESTIONS))
    n_test = int(len(QUESTIONS) * test_frac)
    te, tr = idx[:n_test], idx[n_test:]

    def build(ids):
        f = np.array([featurize(QUESTIONS[i][0])[0] for i in ids])
        y = np.array([QUESTIONS[i][1] for i in ids])
        return f, y

    return build(tr) + build(te)


def route(net, question: str) -> tuple[int | None, float]:
    """Return (intent, confidence); intent None => fallback."""
    from .blocknet import encode_input

    f, total_hits = featurize(question)
    if total_hits == 0:
        return None, 0.0
    logits = net.forward(encode_input(f[None, :]))[:, 0]
    z = np.exp(logits - logits.max())
    p = z / z.sum()
    best = int(np.argmax(p))
    if p[best] < CONFIDENCE_FLOOR:
        return None, float(p[best])
    return best, float(p[best])
