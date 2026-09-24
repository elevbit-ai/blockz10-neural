// Cross-language test: the JS mirror must reproduce the Python model
// bit-for-bit (same weights => same logits) and train on its own.
// Run: node tests/test_blocknet.mjs
import { readFileSync } from "node:fs";
import {
  BlockNet, encodeInput, patternData, patternFeatures, xorData,
} from "../src/js/blocknet.js";

let passed = 0;
const ok = (cond, name) => {
  if (!cond) { console.error(`FAIL  ${name}`); process.exit(1); }
  passed++; console.log(`  ok  ${name}`);
};

const dir = new URL(".", import.meta.url);
const vector = JSON.parse(readFileSync(new URL("./vector.json", dir)));
const model = JSON.parse(
  readFileSync(new URL("../docs/assets/models/xor.json", dir)));

// --- Python model reproduced in JS -----------------------------------------
const net = BlockNet.fromDict(model);
const { logits } = net.forward(encodeInput(vector.inputs));
let worst = 0;
for (let c = 0; c < logits.length; c++)
  for (let b = 0; b < logits[c].length; b++)
    worst = Math.max(worst, Math.abs(logits[c][b] - vector.logits[c][b]));
ok(worst < 1e-9, `JS reproduces Python logits (max diff ${worst.toExponential(1)})`);
ok(JSON.stringify(net.predict(vector.inputs)) === JSON.stringify(vector.predictions),
  "JS reproduces Python predictions");

// --- conservation ------------------------------------------------------------
const Ts = net.matrices();
let colErr = 0;
for (const T of Ts)
  for (let j = 0; j < 16; j++) {
    let s = 0;
    for (let i = 0; i < 16; i++) s += T[i][j];
    colErr = Math.max(colErr, Math.abs(s - 1));
  }
ok(colErr < 1e-12, "every JS transition column sums to 1");

// --- JS trains XOR from scratch -----------------------------------------------
const { features, labels } = xorData();
const fresh = new BlockNet(2, 3, 1.0, 155);
await fresh.train(features, labels, { steps: 2000, lr: 0.6 });
ok(fresh.accuracy(features, labels) === 1, "JS trains XOR to 100% from scratch");

// --- JS trains patterns ---------------------------------------------------------
const tr = patternData(60, 64, 155);
const te = patternData(30, 64, 7);
const pnet = new BlockNet(3, 3, 1.0, 155);
await pnet.train(tr.features, tr.labels, { steps: 1500, lr: 0.6 });
const acc = pnet.accuracy(te.features, te.labels);
ok(acc >= 0.95, `JS trains patterns to >= 95% (got ${(acc * 100).toFixed(1)}%)`);

// --- feature parity ---------------------------------------------------------------
ok(Math.abs(patternFeatures("eee11")[3] - 3 / 5) < 1e-12,
  "Blockz10 compression feature: eee11 -> 311 (3/5)");

// --- save/load ---------------------------------------------------------------------
const clone = BlockNet.fromDict(fresh.toDict());
const a = fresh.forward(encodeInput(features)).logits;
const b = clone.forward(encodeInput(features)).logits;
let same = true;
for (let c = 0; c < 2; c++)
  for (let k = 0; k < 4; k++) if (a[c][k] !== b[c][k]) same = false;
ok(same, "toDict/fromDict reproduces logits exactly");

console.log(`\n${passed} tests passed.`);
