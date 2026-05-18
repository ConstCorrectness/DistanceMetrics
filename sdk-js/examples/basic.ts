import { query_intent } from "@botcierge/sdk";

const result = await query_intent("I have food to share");
console.log(`Domain:     ${result.domain}`);
console.log(`Intent:     ${result.intent}`);
console.log(`Confidence: ${result.confidence}`);
