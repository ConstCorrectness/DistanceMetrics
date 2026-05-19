import * as readline from 'readline';
import { query_intent } from './index';


const queries = [
    "I'm hungry",
    "I need some blood",
    "I want to add milk to my shopping list"
];

async function runBatch() {
    for (const query of queries) {
        const result = await query_intent(query);
        console.log(result.intent);
    }
}

async function runInteractive() {
    const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
    const prompt = () => rl.question('> ', async (user_query: string) => {
        if (!user_query) { rl.close(); return; }
        const result = await query_intent(user_query);
        console.log(result.intent);
        prompt();
    });
    prompt();
}

const mode = process.argv[2];
if (mode === 'interactive') {
    runInteractive();
} else {
    runBatch();
}
