// Cross-language test: JS featurizer + router must match Python exactly.
// Run: node tests/test_chatbot.mjs
import { readFileSync } from "node:fs";
import { BlockNet, encodeInput } from "../src/js/blocknet.js";
import { featurize, routeQuestion } from "../docs/assets/chatbot.js";

let passed = 0;
const ok = (cond, name) => {
  if (!cond) { console.error(`FAIL  ${name}`); process.exit(1); }
  passed++; console.log(`  ok  ${name}`);
};

const dir = new URL(".", import.meta.url);
const vec = JSON.parse(readFileSync(new URL("./chatbot-vector.json", dir), "utf8"));
const net = BlockNet.fromDict(JSON.parse(
  readFileSync(new URL("../docs/assets/models/chatbot.json", dir), "utf8")));

let worst = 0;
for (const v of vec) {
  const { features, totalHits } = featurize(v.q);
  ok(totalHits === v.hits, `hits match Python for ${JSON.stringify(v.q)}`);
  for (let i = 0; i < 10; i++)
    worst = Math.max(worst, Math.abs(features[i] - v.features[i]));
  const r = routeQuestion(net, encodeInput, v.q);
  ok(r.intent === v.intent, `route matches Python (${v.intent}) for ${JSON.stringify(v.q)}`);
  if (v.intent !== null)
    ok(Math.abs(r.confidence - v.confidence) < 1e-9,
      `confidence matches Python (${v.confidence.toFixed(4)})`);
}
ok(worst < 1e-12, `feature vectors match Python (max diff ${worst.toExponential(1)})`);

console.log(`\n${passed} tests passed.`);
