import { Botcierge } from "@botcierge/sdk";

const client = new Botcierge({ baseUrl: "http://staging.botcierge.com:8000" });

const result = await client.query_intent("I need a small loan");
console.log(result);
