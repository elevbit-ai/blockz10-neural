<div align="center">

# «Blockz10 Neural»

### A neural network built only from blocks
*Uma rede neural construída somente de blocos*

**[🌐 Site + laboratório ao vivo](https://elevbit-ai.github.io/blockz10-neural/) · [🎬 Vídeo](https://elevbit-ai.github.io/blockz10-neural/#video) · [📜 Especificação](SPECIFICATION.md) · [🧩 Blockz10](https://github.com/elevbit-ai/blockz10) · [⛓ Block155Learn](https://github.com/elevbit-ai/block155-contract)**

![Python](https://img.shields.io/badge/Python-numpy%20puro-81d4fa?style=flat-square&logo=python&logoColor=white&labelColor=060806)
![JavaScript](https://img.shields.io/badge/JavaScript-zero%20deps-81d4fa?style=flat-square&logo=javascript&logoColor=white&labelColor=060806)
![Solidity](https://img.shields.io/badge/Solidity-0.8.24-81d4fa?style=flat-square&logo=solidity&logoColor=white&labelColor=060806)
![Tests](https://img.shields.io/badge/tests-78%2F78%20passing-00e676?style=flat-square&labelColor=060806)
![Gradcheck](https://img.shields.io/badge/gradcheck-1.4e--10-00e676?style=flat-square&labelColor=060806)
![License](https://img.shields.io/badge/license-MIT-00e676?style=flat-square&labelColor=060806)

Criado por **Joaquim Pedro de Morais Filho** · j360074@hotmail.com · 2020–2026

</div>

---

## O que é / What it is

**PT** — Uma rede neural cujas camadas são a pirâmide **Block 15/5** do
sistema [Blockz10](https://github.com/elevbit-ai/blockz10). As duas
operações originais do modelo viram as duas operações de uma rede
neural: a **redistribuição** entre blocos é a camada linear
(conservativa: colunas do softmax somam 1, valor nunca é criado nem
destruído) e o **bônus por limiar** é a não-linearidade (o mesmo
"joelho" do ReLU). A entrada é um depósito nos blocos; após 3 rodadas,
a classe é lida onde o valor aterrissa na base. Treina por
retropropagação manual em numpy puro — verificada por diferenças
finitas — e roda espelhada em JavaScript no navegador.

**EN** — A neural network whose layers are the **Block 15/5** pyramid
of the [Blockz10](https://github.com/elevbit-ai/blockz10) system. The
model's two original operations become the two operations of a neural
net: **redistribution** between blocks is the linear layer
(conservative: softmax columns sum to 1, value is never created or
destroyed) and the **threshold bonus** is the nonlinearity (the same
kink as ReLU). The input is a deposit into the blocks; after 3 rounds,
the class is read where value lands at the base. It trains by manual
backpropagation in pure numpy — verified against finite differences —
and runs mirrored in JavaScript in the browser.

## A linha evolutiva / The lineage

```
2022  Block 15/5        regras fixas de redistribuição      (o manuscrito)
2026  Block155Learn     pesos aprendíveis on-chain           (o contrato)
2026  Blockz10 Neural   a pirâmide classifica                (este repositório)
```

## Resultados / Results

| Tarefa / Task | O que prova / What it proves | Teste / Test |
|---|---|---|
| **XOR** | não separável por reta → a não-linearidade funciona / not linearly separable → the nonlinearity works | **100%** |
| **Íris** (Fisher, 1936) | aprende dados reais que nunca viu / learns real unseen data | **93,3%** (28/30) |
| **Padrões {e,1}** | reconhece estrutura via a própria codificação Blockz10 / recognizes structure through the Blockz10 encoding itself | **100%** |

~773 parâmetros. Os 2 erros do Íris são os exemplares classicamente
ambíguos versicolor/virginica. Todos os números saem de
`python tests/test_blocknet.py`. / ~773 parameters. Every number comes
from the test suite.

## Experimente ao vivo / Try it live

O site treina a rede **no seu navegador** — perda caindo em tempo real,
depois você interroga a pirâmide (bits do XOR, medidas da flor,
strings {e,1}) e vê o valor descer até a base a cada rodada:
**[elevbit-ai.github.io/blockz10-neural](https://elevbit-ai.github.io/blockz10-neural/)**

## Chat roteado pela pirâmide / Chat routed by the pyramid

O site inclui um assistente **honesto por construção**: as respostas são
pré-escritas (sobre o ecossistema Blockz10) e quem escolhe qual usar é a
rede neural de blocos — a pergunta vira 10 features de famílias de
palavras-chave, a pirâmide roda 3 rodadas e a intenção é o bloco da base
onde o valor aterrissa (96% de acurácia em held-out; sem reconhecimento
→ fallback explícito, a rede nunca inventa conteúdo). / The site ships
an assistant **honest by construction**: answers are pre-written and the
block network chooses which one to use — the question becomes 10
keyword-family features, the pyramid runs 3 rounds and the intent is the
base block where value lands (96% held-out accuracy; no recognition →
explicit fallback, the network never invents content).

## Inferência on-chain / On-chain inference

[`contracts/BlockNetInference.sol`](contracts/BlockNetInference.sol) —
a mesma rede em **ponto fixo puro** (BASE = 10⁹): o construtor **prova a
conservação no deploy** (toda coluna de pesos precisa somar exatamente
10⁹ ou o deploy reverte), a poeira de arredondamento vai para o bloco de
origem (a regra do Block155Splitter) e qualquer coeficiente é
inspecionável on-chain. O quantizador Python
(`src/blockz10_neural/quantize.py`) reproduz o contrato **bit a bit** —
14 testes Foundry com 10 vetores de paridade, e a quantização mantém
100% de concordância com a rede float nas três tarefas (~957k gas por
inferência). / The same network in **pure fixed point** (BASE = 10⁹):
the constructor **proves conservation at deploy time** (every weight
column must sum to exactly 10⁹ or deployment reverts), rounding dust
goes to the origin block (the Block155Splitter rule) and every
coefficient is inspectable on-chain. The Python quantizer reproduces the
contract **bit for bit** — 14 Foundry tests with 10 parity vectors, and
quantization keeps 100% agreement with the float network on all three
tasks (~957k gas per inference).

## Uso rápido / Quick start

```bash
git clone https://github.com/elevbit-ai/blockz10-neural
cd blockz10-neural
python tests/test_blocknet.py   # 17 tests passed.
python tests/test_chatbot.py    # 14 tests passed.
python tests/test_quantize.py   #  8 tests passed.
node tests/test_blocknet.mjs    #  7 tests passed (cross-language vector).
node tests/test_chatbot.mjs     # 18 tests passed (router parity).
forge test                      # 14 tests passed (on-chain parity).
```

```python
>>> import sys; sys.path.insert(0, "src")
>>> from blockz10_neural import BlockNet, iris, xor
>>> f, y = xor()
>>> net = BlockNet(n_classes=2, rounds=3)
>>> net.train(f, y, steps=2000, lr=0.6)
>>> net.accuracy(f, y)
1.0
```

```js
import { BlockNet, xorData } from "./src/js/blocknet.js";
const { features, labels } = xorData();
const net = new BlockNet(2, 3);
await net.train(features, labels, { steps: 2000, lr: 0.6 });
net.accuracy(features, labels);   // 1
```

## Como funciona / How it works

```
entrada (features 0..1)                     16 blocos: origem + 5 níveis
   │  depósito normalizado (Σ = 1)                     [0]
   ▼                                                   [1]
┌──────────── rodada × 3 ────────────┐               [2][3]
│ u  = T·x     T = softmax-coluna(θ) │             [4][5][6]
│               Σ coluna = 1 (conserva)│          [7][8][9][10]
│ x′ = u + g·relu(u − média(u))       │        [11][12][13][14][15]
│               bônus por limiar      │         ▲ leitura da classe
└─────────────────────────────────────┘
classe = argmax(escala·x_base + viés)
```

Retropropagação manual através do softmax por coluna, do bônus (com a
dependência da média) e da leitura — erro máximo contra diferenças
finitas: **1,4 × 10⁻¹⁰**. O espelho JS reproduz os logits do Python com
diferença **3,6 × 10⁻¹⁵**. / Manual backprop through the column-softmax,
the bonus (mean dependency included) and the readout — max error against
finite differences: **1.4 × 10⁻¹⁰**. The JS mirror reproduces Python's
logits within **3.6 × 10⁻¹⁵**.

## Estrutura / Layout

```
blockz10-neural/
├── src/blockz10_neural/    # referência Python (numpy puro)
│   ├── blocknet.py         # rede: forward, backprop manual, treino
│   ├── tasks.py            # XOR · Íris · padrões {e,1}
│   ├── chatbot.py          # roteador de chat (10 famílias → 5 intenções)
│   └── quantize.py         # ponto fixo 1e9, espelho bit a bit do contrato
├── src/js/blocknet.js      # espelho JavaScript (ES module, zero deps)
├── contracts/              # BlockNetInference.sol — inferência on-chain
├── test/                   # 14 testes Foundry (paridade Python → EVM)
├── data/iris.csv           # Fisher (1936), domínio público
├── tests/                  # 64 testes Python/JS (gradcheck, vetores cruzados)
├── docs/                   # site: laboratório + chat + vídeo
├── media/                  # vídeo explicativo + poster
└── tools/                  # train_models · train_chatbot · export_contract · make_video
```

## Limites honestos / Honest limits

16 blocos resolvem tarefas pequenas — é o ponto: demonstrar que **as
regras do Blockz10 são suficientes para aprender**, com a matemática
auditável linha a linha. Não é um modelo de linguagem, e não afirma
ser. / 16 blocks solve small tasks — that is the point: to demonstrate
that **the Blockz10 rules are sufficient to learn**, with the math
auditable line by line. It is not a language model, and does not claim
to be.

## Autor / Author

**Joaquim Pedro de Morais Filho**
📧 j360074@hotmail.com

Criador do sistema Blockz10 (2020) e do modelo Block 15/5 (2022),
registrados on-chain como NFT (coleção Blockz10 · OpenSea) e
documentados desde a origem em
[blockz10.blogspot.com](https://blockz10.blogspot.com). Todo o
conteúdo, conceito e autoria pertencem exclusivamente ao autor.

## Licença / License

[MIT](LICENSE) © 2020–2026 Joaquim Pedro de Morais Filho
