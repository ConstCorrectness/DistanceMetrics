import { query_intent } from "../src/index.ts";

try {
  const result = await query_intent("I want to add an item to my shopping list");
  console.log(`${result.intent} (${result.confidence})`);
} catch (err) {
  console.error("Classification failed:", (err as Error).message);
}
