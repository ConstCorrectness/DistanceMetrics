# @horribleprogram/sdk

[![npm version](https://img.shields.io/npm/v/@horribleprogram/sdk.svg)](https://www.npmjs.com/package/@horribleprogram/sdk)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official JavaScript/TypeScript SDK for **Botcierge Intent Classification**. 

Starting with `v0.1.5`, the SDK runs **completely locally** on your machine. It loads the `sentence-transformers/all-MiniLM-L12-v2` model using ONNX runtime and performs classification via cosine similarity against a built-in static intents taxonomy.

No API keys, configuration, or active internet connection (after first run) are required.

## Features

- **Purely Local**: High-speed offline intent classification with zero external API calls or latency.
- **ONNX-Powered**: Powered by `@xenova/transformers` for running `all-MiniLM-L12-v2` locally.
- **TypeScript Native**: Full type safety for class instances, options, and classification outputs.
- **Automatic Caching**: Model weights (approx. 120MB) are downloaded automatically on the first run and cached locally.

## Installation

```bash
npm install @horribleprogram/sdk
```

## Quick Start

```typescript
import { query_intent } from '@horribleprogram/sdk';

// On the first run, the local model will download and cache.
const result = await query_intent("I'd like to share some food");

console.log(result.domain);     // FOODLINK
console.log(result.intent);     // share_food
console.log(result.confidence); // high
```

## Usage

### Simple Usage
The simplest way to use the SDK is through the default helper function:

```javascript
import { query_intent } from '@horribleprogram/sdk';

const result = await query_intent("I'm hungry");
console.log(result);
// { domain: 'FOODLINK', intent: 'request_food', confidence: 'high', scores: { request_food: 0.975 } }
```

### Dedicated Instance
For class-based architectures, instantiate the `Botcierge` class:

```javascript
import { Botcierge } from '@horribleprogram/sdk';

const client = new Botcierge();
const result = await client.query_intent("I want to add milk to my list");
console.log(result);
// { domain: 'SHOP_SAVVY', intent: 'add_item', confidence: 'high', scores: { add_item: 0.884 } }
```

## API Reference

### `query_intent(utterance: string): Promise<IntentResult>`
Classifies a string using the default local instance.

### `class Botcierge`
#### `constructor(config?: BotciergeConfig)`
Creates a new intent classification instance.
- `config`: Currently empty (reserved for future options).

#### `query_intent(utterance: string): Promise<IntentResult>`
Classifies a string locally into a specific domain and intent.

### Types

#### `IntentResult`
```typescript
interface IntentResult {
  domain: string;
  intent: string;
  confidence: "high" | "medium" | "low";
  scores?: Record<string, number>;
}
```

## Development

To build and test the SDK locally:

```bash
# Clone the repository and install dependencies
cd sdk-js
npm install

# Run unit tests
npm test

# Build files (CJS + ESM + DTS)
npm run build
```

## License

MIT © [horribleprogram](https://npmjs.com/~horribleprogram)

