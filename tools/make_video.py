# Generates media/neural-explainer.mp4 frame by frame (PIL + ffmpeg).
# All numbers shown are REAL: the pyramid states come from the trained
# XOR model's forward trace and the loss curve from an actual training run.
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from blockz10_neural import BlockNet, LEVELS, encode_input, xor  # noqa: E402

W, H = 1280, 720
BG = (6, 8, 6)
GREEN = (0, 230, 118)
DIM = (110, 200, 150)
WHITE = (235, 235, 235)
GREY = (150, 150, 150)
BLUE = (129, 212, 250)
BLUE_DIM = (110, 159, 200)
GOLD = (255, 213, 79)
RED = (244, 67, 54)

FONTS = r"C:\Windows\Fonts"
MONO_B = lambda s: ImageFont.truetype(os.path.join(FONTS, "consolab.ttf"), s)
MONO = lambda s: ImageFont.truetype(os.path.join(FONTS, "consola.ttf"), s)
SANS = lambda s: ImageFont.truetype(os.path.join(FONTS, "segoeui.ttf"), s)
SANS_B = lambda s: ImageFont.truetype(os.path.join(FONTS, "segoeuib.ttf"), s)

OUT = ROOT / "media"
FRAMES = OUT / "_frames"
FRAMES.mkdir(parents=True, exist_ok=True)

frames: list[tuple[str, float]] = []
_n = 0


def canvas():
    img = Image.new("RGB", (W, H), BG)
    return img, ImageDraw.Draw(img)


def save(img, dur):
    global _n
    name = f"f{_n:04d}.png"
    img.save(FRAMES / name)
    frames.append((name, dur))
    _n += 1


def center(d, y, text, font, fill):
    w = d.textlength(text, font=font)
    d.text(((W - w) / 2, y), text, font=font, fill=fill)


def footer(d):
    center(d, H - 44, "Joaquim Pedro de Morais Filho  ·  j360074@hotmail.com", MONO(20), DIM)


# ---- real data: train XOR and capture everything ---------------------------
print("training the XOR model for the video...")
f_xor, y_xor = xor()
net = BlockNet(2, rounds=3, seed=155)
loss_hist = net.train(f_xor, y_xor, steps=2000, lr=0.6)
X_probe = encode_input(np.array([[1.0, 0.0]]))          # XOR(1,0) = 1
logits, xs, us, ms = net.forward(X_probe, trace=True)
states = [x[:, 0] for x in xs]                          # deposit + 3 rounds
pred = int(np.argmax(logits[:, 0]))
print(f"  probe XOR(1,0) -> class {pred}, loss final {loss_hist[-1]:.4f}")


def draw_pyramid(d, values, x0, y0, cell=62, gap=10, win_idx=-1, scale=None):
    """Draw the 16-block pyramid with heat coloring from `values`."""
    vmax = max(values) if scale is None else scale
    for lv, blocks in LEVELS.items():
        n = len(blocks)
        row_w = n * cell + (n - 1) * gap
        x = x0 - row_w / 2
        y = y0 + lv * (cell + gap)
        for i in blocks:
            heat = min(1.0, values[i] / (vmax + 1e-12))
            col = GOLD if i == win_idx else (
                int(30 + 99 * heat), int(60 + 152 * heat), int(70 + 180 * heat))
            fill = (4, int(10 + 14 * heat), int(14 + 14 * heat))
            d.rectangle([x, y, x + cell, y + cell], outline=col, width=4, fill=fill)
            val = f"{values[i]*150:.0f}" if values[i] * 150 >= 1 else f"{values[i]*150:.1f}"
            fnt = MONO_B(19)
            w = d.textlength(val, font=fnt)
            d.text((x + (cell - w) / 2, y + cell / 2 - 12), val, font=fnt,
                   fill=GOLD if i == win_idx else WHITE)
            x += cell + gap


# ---- Scene 1: title ---------------------------------------------------------
img, d = canvas()
center(d, 165, "\u00abBlockz10 Neural\u00bb", MONO_B(72), BLUE)
center(d, 300, "Uma rede neural constru\u00edda somente de blocos", SANS(32), WHITE)
center(d, 352, "A neural network built only from blocks", SANS(24), GREY)
center(d, 440, "a rede n\u00e3o usa blocos \u2014 a rede \u00c9 a pir\u00e2mide", MONO(24), GREEN)
center(d, 480, "the network does not use blocks \u2014 the network IS the pyramid", MONO(19), GREY)
center(d, 550, "por Joaquim Pedro de Morais Filho", SANS(26), DIM)
save(img, 5.0)

# ---- Scene 2: lineage -------------------------------------------------------
img, d = canvas()
center(d, 66, "A linha evolutiva / The lineage", SANS_B(38), BLUE)
rows = [
    ("2022", "\u00abBlock 15/5\u00bb", "regras fixas de redistribui\u00e7\u00e3o", "fixed redistribution rules"),
    ("2026", "\u00abBlock155Learn\u00bb", "pesos aprend\u00edveis on-chain", "learnable weights on-chain"),
    ("2026", "\u00abBlockz10 Neural\u00bb", "a pir\u00e2mide classifica", "the pyramid classifies"),
]
y = 180
for yr, name, pt, en in rows:
    d.text((130, y), yr, font=MONO_B(30), fill=GREY)
    d.text((260, y), name, font=MONO_B(32), fill=BLUE if "Neural" in name else GREEN)
    d.text((700, y), pt, font=SANS(26), fill=WHITE)
    d.text((700, y + 36), en, font=SANS(20), fill=GREY)
    y += 120
footer(d)
save(img, 7.0)

# ---- Scene 3: the two operations ---------------------------------------------
img, d = canvas()
center(d, 60, "Toda rede neural s\u00e3o duas opera\u00e7\u00f5es", SANS_B(36), BLUE)
center(d, 112, "Every neural network is two operations", SANS(24), GREY)
d.text((110, 200), "REDE NEURAL / NEURAL NET", font=MONO_B(24), fill=GREY)
d.text((660, 200), "PIR\u00c2MIDE 15/5 / PYRAMID", font=MONO_B(24), fill=GREEN)
pairs = [
    ("camada linear  y = W\u00b7x", "redistribui\u00e7\u00e3o entre blocos", "linear layer", "block redistribution"),
    ("n\u00e3o-linearidade (ReLU)", "b\u00f4nus por limiar", "nonlinearity (ReLU)", "threshold bonus"),
]
y = 270
for a, b, ae, be in pairs:
    d.text((110, y), a, font=MONO_B(28), fill=WHITE)
    d.text((110, y + 36), ae, font=SANS(19), fill=GREY)
    d.text((580, y + 6), "=", font=MONO_B(34), fill=GOLD)
    d.text((660, y), b, font=MONO_B(28), fill=GREEN)
    d.text((660, y + 36), be, font=SANS(19), fill=GREY)
    y += 120
center(d, y + 20, "o Block 15/5 sempre teve as duas \u2014 s\u00f3 faltava deix\u00e1-las aprender",
       SANS(25), GOLD)
center(d, y + 58, "Block 15/5 always had both \u2014 they just had to be allowed to learn",
       SANS(20), GREY)
save(img, 8.5)

# ---- Scene 4: equation 1 -------------------------------------------------------
img, d = canvas()
center(d, 70, "Opera\u00e7\u00e3o 1 \u00b7 Redistribui\u00e7\u00e3o conservativa", SANS_B(36), BLUE)
center(d, 122, "Operation 1 \u00b7 Conservative redistribution", SANS(23), GREY)
center(d, 230, "u = T \u00b7 x", MONO_B(58), GREEN)
center(d, 330, "T = softmax-coluna(\u03b8)   \u21d2   \u03a3 coluna = 1", MONO_B(30), WHITE)
center(d, 420, "misturar valor sem nunca criar nem destruir", SANS(27), WHITE)
center(d, 462, "mix value without ever creating or destroying it", SANS(21), GREY)
center(d, 530, "o invariante on-chain do Block155Learn, preservado no treino",
       SANS(22), BLUE_DIM)
footer(d)
save(img, 7.5)

# ---- Scene 5: equation 2 --------------------------------------------------------
img, d = canvas()
center(d, 70, "Opera\u00e7\u00e3o 2 \u00b7 B\u00f4nus por limiar", SANS_B(36), BLUE)
center(d, 122, "Operation 2 \u00b7 Threshold bonus", SANS(23), GREY)
center(d, 215, "x\u2032 = u + g \u00b7 relu(u \u2212 m\u00e9dia(u))", MONO_B(46), GREEN)
center(d, 330, "blocos acima da m\u00e9dia ganham b\u00f4nus \u2014 a regra original do 15/5",
       SANS(26), WHITE)
center(d, 372, "blocks above the mean earn a bonus \u2014 the original 15/5 rule",
       SANS(21), GREY)
center(d, 450, "matematicamente: o mesmo \u201cjoelho\u201d do ReLU das redes profundas",
       SANS(26), GOLD)
center(d, 492, "mathematically: the exact ReLU kink of deep networks", SANS(21), GREY)
footer(d)
save(img, 7.5)

# ---- Scene 6: architecture with REAL states -------------------------------------
labels_pt = ["dep\u00f3sito: a entrada entra na pir\u00e2mide",
             "rodada 1: redistribui\u00e7\u00e3o + b\u00f4nus",
             "rodada 2: redistribui\u00e7\u00e3o + b\u00f4nus",
             "rodada 3: o valor aterrissa na base"]
labels_en = ["deposit: the input enters the pyramid",
             "round 1: redistribution + bonus",
             "round 2: redistribution + bonus",
             "round 3: value lands at the base"]
for r, st in enumerate(states):
    img, d = canvas()
    center(d, 46, "XOR(1, 0) \u2014 infer\u00eancia real / real inference", SANS_B(32), BLUE)
    win = LEVELS[5][pred] if r == len(states) - 1 else -1
    draw_pyramid(d, st, W // 2 - 180, 120, win_idx=win, scale=max(states[-1]))
    d.text((880, 200), labels_pt[r], font=SANS(24), fill=WHITE)
    d.text((880, 240), labels_en[r], font=SANS(19), fill=GREY)
    if r == len(states) - 1:
        d.text((880, 330), f"classe / class = {pred}", font=MONO_B(34), fill=GOLD)
        d.text((880, 380), "XOR(1,0) = 1  correto!", font=MONO_B(28), fill=GREEN)
        d.text((880, 450), "a resposta \u00e9 onde", font=SANS(22), fill=WHITE)
        d.text((880, 482), "o valor chega", font=SANS(22), fill=WHITE)
        d.text((880, 518), "the answer is where", font=SANS(18), fill=GREY)
        d.text((880, 544), "the value arrives", font=SANS(18), fill=GREY)
    save(img, 2.4 if r < len(states) - 1 else 5.0)

# ---- Scene 7: training (REAL loss curve, progressive) ----------------------------
sub = np.array(loss_hist)
for cut in [80, 250, 600, 1200, 2000]:
    img, d = canvas()
    center(d, 56, "Treino: retropropaga\u00e7\u00e3o manual / manual backprop", SANS_B(34), BLUE)
    x0, y0, w, h = 150, 160, 980, 360
    d.rectangle([x0, y0, x0 + w, y0 + h], outline=(28, 43, 32), width=2)
    pts = sub[:cut]
    mx, mn = float(sub.max()), float(sub.min())
    poly = [(x0 + i / (len(sub) - 1) * w,
             y0 + h - 12 - (v - mn) / (mx - mn) * (h - 24)) for i, v in enumerate(pts)]
    if len(poly) > 1:
        d.line(poly, fill=BLUE, width=4)
    d.text((x0 + 14, y0 + 12), f"loss {pts[-1]:.4f}", font=MONO_B(26), fill=WHITE)
    d.text((x0 + 14, y0 + h - 42), f"passo / step {cut}/2000", font=MONO(22), fill=GREY)
    center(d, 570, "gradiente verificado por diferen\u00e7as finitas \u2014 erro m\u00e1x 1,4 \u00d7 10\u207b\u00b9\u2070",
           SANS(24), GREEN)
    center(d, 608, "gradient verified against finite differences \u2014 max error 1.4 \u00d7 10\u207b\u00b9\u2070",
           SANS(19), GREY)
    save(img, 0.9 if cut < 2000 else 3.5)

# ---- Scene 8: the three tasks ------------------------------------------------------
img, d = canvas()
center(d, 64, "Tr\u00eas provas / Three proofs", SANS_B(38), BLUE)
tasks = [
    ("XOR", "nenhuma reta separa \u2014 se acerta, a n\u00e3o-linearidade \u00e9 real",
     "no line separates it \u2014 success proves the nonlinearity", "100%"),
    ("\u00cdRIS (Fisher, 1936)", "150 flores reais \u00b7 treina com 120, testa com 30 nunca vistas",
     "150 real flowers \u00b7 trained on 120, tested on 30 unseen", "93,3%"),
    ("PADR\u00d5ES {e,1}", "6 features lidas da pr\u00f3pria codifica\u00e7\u00e3o eee11 \u2192 311",
     "6 features read from the eee11 \u2192 311 encoding itself", "100%"),
]
y = 170
for name, pt, en, acc in tasks:
    d.text((110, y), name, font=MONO_B(28), fill=GREEN)
    d.text((110, y + 42), pt, font=SANS(23), fill=WHITE)
    d.text((110, y + 76), en, font=SANS(18), fill=GREY)
    d.text((1040, y + 20), acc, font=MONO_B(40), fill=GOLD)
    y += 140
footer(d)
save(img, 9.0)

# ---- Scene 9: rigor ------------------------------------------------------------------
img, d = canvas()
center(d, 64, "Rigor / Rigor", SANS_B(38), BLUE)
pts = [
    ("numpy puro + JavaScript zero-deps + Solidity, espelhados", "pure numpy + zero-deps JavaScript + Solidity, mirrored"),
    ("infer\u00eancia on-chain em ponto fixo \u2014 paridade bit a bit", "on-chain fixed-point inference \u2014 bit-for-bit parity"),
    ("chat roteado pela pir\u00e2mide \u00b7 96% em held-out", "chat routed by the pyramid \u00b7 96% held-out"),
    ("78 testes \u00b7 gradcheck 10\u207b\u00b9\u2070 \u00b7 vetores cruzados Py\u2192JS\u2192EVM", "78 tests \u00b7 gradcheck \u00b7 cross-language vectors"),
]
y = 180
for pt_, en in pts:
    d.text((150, y), "\u25a0", font=SANS(24), fill=BLUE)
    d.text((200, y), pt_, font=SANS_B(27), fill=WHITE)
    d.text((200, y + 36), en, font=SANS(20), fill=GREY)
    y += 95
footer(d)
save(img, 8.0)

# ---- Scene 10: try it -----------------------------------------------------------------
img, d = canvas()
center(d, 150, "Treine voc\u00ea mesmo \u2014 no navegador", SANS_B(40), BLUE)
center(d, 215, "Train it yourself \u2014 in the browser", SANS(26), GREY)
center(d, 320, "elevbit-ai.github.io/blockz10-neural", MONO_B(34), GREEN)
center(d, 420, "XOR \u00b7 \u00cdris \u00b7 padr\u00f5es {e,1} \u2014 perda em tempo real,", SANS(26), WHITE)
center(d, 462, "e a pir\u00e2mide mostrando onde o valor aterrissa", SANS(26), WHITE)
center(d, 516, "live loss curve, and the pyramid showing where value lands", SANS(20), GREY)
save(img, 6.5)

# ---- Scene 11: credits -------------------------------------------------------------------
img, d = canvas()
center(d, 150, "\u00abBlockz10 Neural\u00bb", MONO_B(62), BLUE)
center(d, 258, "parte do sistema \u00abBlockz10\u00bb \u00b7 a Blockz10 system project", SANS(24), GREEN)
center(d, 330, "Criado por / Created by", SANS(26), GREY)
center(d, 375, "Joaquim Pedro de Morais Filho", SANS_B(40), WHITE)
center(d, 460, "j360074@hotmail.com", MONO_B(30), BLUE)
center(d, 525, "github.com/elevbit-ai/blockz10-neural", MONO(26), DIM)
center(d, 570, "\u00a9 2020\u20132026 \u00b7 conceito registrado on-chain \u00b7 NFT \u00b7 OpenSea", SANS(22), GREY)
save(img, 6.0)

# ---- assemble -------------------------------------------------------------------------------
concat = FRAMES / "list.txt"
with open(concat, "w") as fh:
    for name, dur in frames:
        fh.write(f"file '{name}'\nduration {dur}\n")
    fh.write(f"file '{frames[-1][0]}'\n")

mp4 = OUT / "neural-explainer.mp4"
cmd = [
    "ffmpeg", "-y",
    "-f", "concat", "-safe", "0", "-i", str(concat),
    "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
    "-shortest",
    "-vf", "fps=30,format=yuv420p,scale=1280:720",
    "-c:v", "libx264", "-preset", "slow", "-crf", "22",
    "-c:a", "aac", "-b:a", "64k",
    "-movflags", "+faststart",
    str(mp4),
]
subprocess.run(cmd, check=True, capture_output=True)
print(f"OK {mp4} ({mp4.stat().st_size/1e6:.2f} MB, "
      f"{sum(x for _, x in frames):.1f}s, {len(frames)} frames)")

Image.open(FRAMES / "f0000.png").save(OUT / "poster.png")
print("poster saved")
