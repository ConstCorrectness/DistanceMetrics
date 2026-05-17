# @botcierge/sdk

[![npm version](https://img.shields.io/npm/v/@botcierge/sdk.svg)](https://www.npmjs.com/package/@botcierge/sdk)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official JavaScript/TypeScript SDK for the **Botcierge Intent Classification API**.

## Features

- **TypeScript Native**: Full type definitions for all API responses.
- **Lightweight**: Zero dependencies (uses native `fetch`).
- **Flexible**: Easy to use for both simple scripts and complex applications.
- **Cross-platform**: Works in Node.js, Browsers, and Edge environments.

## Installation

```bash
npm install @botcierge/sdk
```

## Quick Start

```typescript
import { query_intent } from '@botcierge/sdk';

try {
  const result = await query_intent("I'd like to share some food");
  
  console.log(`Domain: ${result.domain}`);         // foodshare
  console.log(`Intent: ${result.intent}`);         // share_food
  console.log(`Confidence: ${result.confidence}`); // high
} catch (error) {
  console.error("Classification failed:", error.message);
}
```

## API Reference

### `query_intent(utterance: string): Promise<IntentResult>`

A helper function for quick classification using the default configuration (connecting to `http://localhost:8000`).

### `class Botcierge`

The main class for interacting with the Botcierge API.

#### `constructor(config?: BotciergeConfig)`

- `config.baseUrl`: The base URL of your Botcierge API (default: `http://localhost:8000`).

#### `query_intent(utterance: string): Promise<IntentResult>`

Classifies a string into a specific domain and intent.

### Types

#### `IntentResult`
```typescript
interface IntentResult {
  domain: string;                   // The high-level category (e.g., 'moneyshare')
  intent: string;                   // The specific action (e.g., 'request_loan')
  confidence: "high" | "medium" | "low";
  scores: Record<string, number>;   // Raw similarity scores for all intents
}
```

## Error Handling

The SDK throws standard `Error` objects if the network request fails or if the API returns a non-2xx status code.

```typescript
import { Botcierge } from '@botcierge/sdk';

const client = new Botcierge({ baseUrl: 'https://invalid-api.com' });

try {
  await client.query_intent("hello");
} catch (e) {
  console.log(e.message); // "Botcierge API error: ..."
}
```

## Development

### Running Tests
```bash
npm test
```

### Building
```bash
npm run build
```

## License

MIT © [horribleprogram](https://npmjs.com/~horribleprogram)
