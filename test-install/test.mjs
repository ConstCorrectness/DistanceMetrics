import { query_intent } from '@horribleprogram/sdk';
query_intent("I'm hungry").then(r => console.log(JSON.stringify(r))).catch(e => console.error(e.message));
