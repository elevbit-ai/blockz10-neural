# Blockz10 Neural — Especificação formal / Formal specification

**Versão / Version:** 1.0.0 · **Formato de modelo / Model format:** `blockz10-neural/1`
**Autor / Author:** Joaquim Pedro de Morais Filho · j360074@hotmail.com
**Linha / Lineage:** Block 15/5 (regras fixas, 2022) → Block155Learn (pesos aprendíveis on-chain, 2026) → **Blockz10 Neural** (rede que classifica, 2026)

---

## 1 · Tese / Thesis

**PT** — Uma rede neural alterna transformação linear e não-linearidade.
A pirâmide Block 15/5 sempre teve as duas: a **redistribuição** entre
blocos é uma multiplicação por matriz, e o **bônus/perda por limiar** é
uma função por partes — o mesmo "joelho" do ReLU. Este projeto torna as
duas operações aprendíveis e lê a classificação na base da pirâmide.
A rede não usa blocos; a rede **é** a pirâmide.

**EN** — A neural network alternates linear transformation and
nonlinearity. The Block 15/5 pyramid always had both: **redistribution**
between blocks is a matrix multiplication, and the **threshold
bonus/loss** is a piecewise function — the same "kink" as ReLU. This
project makes both operations learnable and reads the classification at
the pyramid's base. The network does not use blocks; the network **is**
the pyramid.

## 2 · Estado / State

16 blocos: origem (índice 0) + 15 blocos em 5 níveis. / 16 blocks:
origin (index 0) + 15 blocks in 5 levels.

```
nível/level 0 :  [0]                 (origem/origin)
nível/level 1 :  [1]
nível/level 2 :  [2, 3]
nível/level 3 :  [4, 5, 6]
nível/level 4 :  [7, 8, 9, 10]
nível/level 5 :  [11, 12, 13, 14, 15]   ← leitura/readout (base)
```

**Entrada / Input** — até 10 features em [0,1] entram nos blocos 1–10;
os demais recebem um piso de 0,02; a coluna é normalizada para somar 1.
Todo exemplo é um **depósito do mesmo total** — a pirâmide sempre
recebe a mesma quantidade de valor, só muda a forma. / Up to 10
features in [0,1] enter blocks 1–10; the rest get a 0.02 floor; the
column is normalized to sum 1. Every example is a **deposit of the same
total** — the pyramid always receives the same amount of value, only
its shape changes.

## 3 · Inferência / Inference

`R = 3` rodadas de duas equações / rounds of two equations:

```
(1)  u = T_r · x            T_r = softmax-coluna(θ_r)   ⇒  Σ_i T[i,j] = 1
(2)  x′ = u + g · relu(u − mean(u))                        g = 1
```

**(1) Redistribuição conservativa** — colunas somam 1, logo
`Σ(T·x) = Σx` sempre: a mistura não cria nem destrói valor. É o
invariante on-chain do Block155Learn (`Σ weights == BASE`), preservado
no treino por construção. / **Conservative redistribution** — columns
sum to 1, so mixing can never create or destroy value. It is
Block155Learn's on-chain invariant, preserved during training by
construction.

**(2) Bônus por limiar** — blocos acima da média da pirâmide ganham
bônus proporcional ao excesso; a regra de bônus do Block 15/5 original.
É a única fonte de não-linearidade e o único ponto onde o total muda —
exatamente como nas regras do manuscrito. / **Threshold bonus** —
blocks above the pyramid mean earn a bonus proportional to the excess;
the original Block 15/5 bonus rule. It is the only source of
nonlinearity and the only point where the total changes — exactly as in
the manuscript rules.

**Leitura / Readout:**

```
(3)  logits_c = escala · x_K[base_c] + viés_c        base = [11..15]
     classe   = argmax(logits)
```

A resposta é onde o valor aterrissa na base. / The answer is where
value lands at the base.

## 4 · Treino / Training

- Perda: entropia cruzada sobre softmax dos logits. / Loss:
  cross-entropy over the softmax of the logits.
- Gradiente: retropropagação **manual** (numpy puro, sem frameworks)
  através de (3), (2) — incluindo a dependência da média — e (1),
  incluindo o jacobiano do softmax por coluna. / Gradient: **manual**
  backpropagation (pure numpy, no frameworks) through (3), (2) —
  including the mean dependency — and (1), including the column-softmax
  jacobian.
- Verificação: diferenças finitas na suíte de testes; erro máximo
  observado **1,4 × 10⁻¹⁰**. / Verification: finite differences in the
  test suite; max observed error **1.4 × 10⁻¹⁰**.
- Otimizador: gradiente descendente com momento 0,9, lote completo. /
  Optimizer: momentum 0.9 gradient descent, full batch.
- Parâmetros: 3 × (16×16) logits θ + escala + viés ≈ **773**. /
  Parameters: 3 × (16×16) θ logits + scale + bias ≈ **773**.

## 5 · Tarefas e resultados / Tasks and results

| Tarefa / Task | Entrada / Input | Classes | Teste / Test |
|---|---|---|---|
| **XOR** | 2 bits | 2 | **100%** (4/4) |
| **Íris** (Fisher, 1936) | 4 medidas, min-max / 4 measurements | 3 | **93,3%** (28/30) |
| **Padrões {e,1}** | 6 features da codificação Blockz10 | 3 | **100%** (150/150) |

- XOR não é separável linearmente: acertar 100% **prova** que o bônus
  por limiar funciona como não-linearidade. / XOR is not linearly
  separable: reaching 100% **proves** the threshold bonus works as a
  nonlinearity.
- Íris: split estratificado 120/30 (semente 155); os 2 erros são
  exemplares classicamente ambíguos versicolor/virginica. / Iris:
  stratified 120/30 split (seed 155); the 2 misses are the classically
  ambiguous versicolor/virginica specimens.
- Padrões: strings {e,1} de 64 símbolos geradas por cadeias de Markov
  (persistência 0,93 / 0,12 / 0,50); as features vêm da estrutura de
  runs que a codificação canônica `eee11 → 311` torna explícita. /
  Patterns: 64-symbol {e,1} strings from Markov chains (persistence
  0.93 / 0.12 / 0.50); features come from the run structure the
  canonical encoding makes explicit.

## 6 · Formato do modelo / Model format

```json
{
  "format": "blockz10-neural/1",
  "n_classes": 3, "rounds": 3, "gain": 1.0,
  "scale": 21.7, "bias": [ ... C ... ],
  "thetas": [ [16×16], [16×16], [16×16] ]
}
```

Python e JavaScript carregam o mesmo JSON; o espelho JS reproduz os
logits do Python com diferença máxima de **3,6 × 10⁻¹⁵** (precisão de
máquina — `tests/vector.json`). / Python and JavaScript load the same
JSON; the JS mirror reproduces Python's logits within **3.6 × 10⁻¹⁵**
(machine precision — `tests/vector.json`).

## 7 · Limites honestos / Honest limits

- 16 blocos ≈ 773 parâmetros: tarefas pequenas, por design. Não é, nem
  pretende ser, um modelo de linguagem. / 16 blocks ≈ 773 parameters:
  small tasks, by design. It is not, and does not claim to be, a
  language model.
- Redes lineares-por-partes são aproximadores universais **no limite de
  largura/profundidade**; a garantia é da família, não deste tamanho. /
  Piecewise-linear networks are universal approximators **in the
  width/depth limit**; the guarantee belongs to the family, not to this
  size.
- Íris a 93,3% está no intervalo esperado para modelos pequenos; números
  maiores nesse dataset costumam exigir mais capacidade ou sorte de
  split. / Iris at 93.3% is in the expected range for small models.

## 8 · Implementações de referência / Reference implementations

| | Python | JavaScript |
|---|---|---|
| Arquivo / File | `src/blockz10_neural/blocknet.py` | `src/js/blocknet.js` |
| Dependências / Deps | numpy | nenhuma / none (ES module) |
| Testes / Tests | `tests/test_blocknet.py` (17) | `tests/test_blocknet.mjs` (7, incl. vetor cruzado / cross vector) |

---

© 2020–2026 Joaquim Pedro de Morais Filho · Licença MIT / MIT License
Conceito Blockz10 registrado on-chain (NFT · OpenSea) / Blockz10 concept registered on-chain (NFT · OpenSea)
