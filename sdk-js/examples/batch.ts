import { Botcierge } from "../src/index.ts";

const client = new Botcierge();

const utterances = [
  "I'm hungry",
  "I want to remember something",
  "I want to buy a package",
  "Create a flow plan for tomorrow",
];

const results = await Promise.all(utterances.map(u => client.query_intent(u)));

results.forEach((r, i) => {
  console.log(`"${utterances[i]}"`);
  console.log(`  → ${r.domain} / ${r.intent} [${r.confidence}]`);
  console.log();
});
