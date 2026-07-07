# @horribleprogram/sdk

[![npm version](https://img.shields.io/npm/v/@horribleprogram/sdk.svg)](https://www.npmjs.com/package/@horribleprogram/sdk)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official JavaScript/TypeScript SDK for **Botcierge Intent Classification** and **Company Embeddings Semantic Search & CRUD**.

Starting with `v0.1.5`, the SDK runs **completely locally** on your machine by default. It loads the `sentence-transformers/all-MiniLM-L12-v2` model using ONNX runtime and performs classification and embedding generation offline.

---

## Features

- **Purely Local Intent Classification**: High-speed offline intent classification with zero external API calls or latency.
- **MongoDB Semantic Search**: Store, manage, and query company records in MongoDB based on semantic similarity.
- **Flexible Embeddings Backend**: Run embedding generation 100% locally with the ONNX-powered `all-MiniLM-L12-v2` model, or switch to OpenAI's `text-embedding-3-small` API for high-quality remote embeddings if configured.
- **Parallel Batch Processing**: Fast bulk CSV seeding with parallelizable embedding generation.
- **TypeScript Native**: Full type safety for configs, company models, and query results.

---

## Installation

```bash
npm install @horribleprogram/sdk
```

---

## Quick Start

### 1. Intent Classification (Local Only)

```typescript
import { query_intent } from '@horribleprogram/sdk';

// On the first run, the local model will download and cache.
const result = await query_intent("I'd like to share some food");

console.log(result.domain);     // FOODLINK
console.log(result.intent);     // share_food
console.log(result.confidence); // high
```

### 2. Company Embeddings & MongoDB CRUD

Manage companies in MongoDB and perform semantic queries over them:

```typescript
import { addCompany, queryCompanies, closeConnection } from '@horribleprogram/sdk';

const config = {
  uri: 'mongodb://localhost:27017',
  dbName: 'distance_metrics',
  collectionName: 'companies',
};

// 1. Add a company (embeddings are computed automatically)
await addCompany({
  company_name: 'Antigravity Code Labs',
  website: 'https://antigravity.ai',
  short_description: 'An advanced AI programming assistant team designed by Google DeepMind.',
  product_description: 'We build autonomous agent systems that help users write, debug, and ship software at scale.',
  mapped_function: 'AI Software Development',
  mapped_industry: 'Developer Tools',
  match_keywords: 'ai, programming, agent, coding, deepmind',
  aliases: 'Antigravity AI',
  active: true,
  priority: 'High',
  source: 'manual',
  zone: 'N/A'
}, config);

// 2. Query companies semantically
const results = await queryCompanies('AI coding assistant by DeepMind', 3, config);

for (const match of results) {
  console.log(`${match.company_name} (Similarity: ${match.similarity.toFixed(4)})`);
  // "Antigravity Code Labs (Similarity: 0.7245)"
}

// 3. Clean up database connection
await closeConnection();
```

---

## Usage & Integration Guides

### Intent Classification
For class-based architectures, instantiate the `Botcierge` class:

```typescript
import { Botcierge } from '@horribleprogram/sdk';

const client = new Botcierge();
const result = await client.query_intent("I want to add milk to my list");
console.log(result);
// { domain: 'SHOP_SAVVY', intent: 'add_item', confidence: 'high', scores: { add_item: 0.884 } }
```

### Company Embedding Management (`CompanyEmbeddingManager`)
For long-lived database operations or custom integration, use the `CompanyEmbeddingManager` class directly.

#### Basic Usage (Local MongoDB)
```typescript
import { CompanyEmbeddingManager } from '@horribleprogram/sdk';

const manager = new CompanyEmbeddingManager({
  uri: 'mongodb://localhost:27017',
  dbName: 'distance_metrics',
  collectionName: 'companies',
  // Use OpenAI instead of local model if key is provided:
  openAIApiKey: process.env.OPENAI_API_KEY, 
  preferOpenAI: true
});

// Perform database operations
const col = await manager.getCollection();
const count = await col.countDocuments();
console.log(`Connected to local database. Document count: ${count}`);

await manager.disconnect();
```

#### MongoDB Atlas Cluster Usage (Secure Cloud Connection)
The SDK natively supports MongoDB Atlas secure clusters via `mongodb+srv://` URIs. Under Node.js, the `mongodb` v7 driver handles DNS resolving, SSL/TLS, and authentication automatically out-of-the-box, with no extra external dependencies or configuration required.

Simply provide your Atlas connection string and specify options like `serverApi` in the `mongoOptions` object (which is passed directly to the internal `MongoClient` constructor):

```typescript
import { CompanyEmbeddingManager } from '@horribleprogram/sdk';

const manager = new CompanyEmbeddingManager({
  uri: 'mongodb+srv://<username>:<password>@cluster0.xxxx.mongodb.net/myDatabase?retryWrites=true&w=majority',
  dbName: 'distance_metrics',
  // Custom options passed directly to the MongoClient constructor:
  mongoOptions: {
    serverApi: {
      version: '1' as any, // ServerApiVersion.v1 (ensures long-term API compatibility)
      strict: true,
      deprecationErrors: true,
    }
  }
});

// Perform operations on MongoDB Atlas
const col = await manager.getCollection();
const count = await col.countDocuments();
console.log(`Connected to MongoDB Atlas! Document count: ${count}`);

await manager.disconnect();
```

#### Seeding Companies from CSV
You can batch-onboard/seed companies directly from a CSV file. It reads the CSV, generates embeddings in parallel batches, and inserts/overwrites the MongoDB collection.

```typescript
import { seedCompanies, closeConnection } from '@horribleprogram/sdk';

async function seed() {
  const count = await seedCompanies('./companies.csv', {
    uri: 'mongodb://localhost:27017',
    dbName: 'distance_metrics',
  });
  console.log(`Seeded ${count} companies!`);
  await closeConnection();
}
```

---

## API Reference

### Intent Classification

#### `query_intent(utterance: string): Promise<IntentResult>`
Classifies a string using the default local instance.

#### `class Botcierge`
* `constructor(config?: BotciergeConfig)`
* `query_intent(utterance: string): Promise<IntentResult>`

---

### Company Embeddings & CRUD

#### `seedCompanies(csvPath: string, config?: MongoEmbeddingConfig): Promise<number>`
Seeds the MongoDB collection with companies from a CSV file. Returns the number of seeded records.

#### `addCompany(company: Omit<Company, "embedding"> & { embedding?: number[] }, config?: MongoEmbeddingConfig): Promise<void>`
Upserts a company record in the database. If no `embedding` is provided, it is automatically computed based on the company's fields.

#### `removeCompany(companyName: string, config?: MongoEmbeddingConfig): Promise<boolean>`
Deletes a company by name (case-insensitive regular expression match). Returns `true` if a record was deleted, `false` otherwise.

#### `queryCompanies(prompt: string, limit?: number, config?: MongoEmbeddingConfig): Promise<(Company & { similarity: number })[]>`
Performs a vector search over the MongoDB database by calculating the cosine similarity of the company embeddings against the input prompt embedding. Returns list of companies sorted descending by `similarity`.

#### `closeConnection(): Promise<void>`
Disconnects and clears the default active database manager.

#### `class CompanyEmbeddingManager`
* `constructor(config?: MongoEmbeddingConfig)`
* `connect(): Promise<MongoClient>`
* `disconnect(): Promise<void>`
* `getCollection(): Promise<Collection<Company>>`
* `getEmbedding(text: string): Promise<number[]>`
* `getEmbeddings(texts: string[]): Promise<number[][]>`
* `addCompany(company: Company): Promise<void>`
* `removeCompany(companyName: string): Promise<boolean>`
* `queryCompanies(prompt: string, limit?: number): Promise<(Company & { similarity: number })[]>`
* `seedFromCSV(csvPath: string): Promise<number>`

---

## Types

#### `IntentResult`
```typescript
interface IntentResult {
  domain: string;
  intent: string;
  confidence: "high" | "medium" | "low";
  scores?: Record<string, number>;
}
```

#### `MongoEmbeddingConfig`
```typescript
interface MongoEmbeddingConfig {
  uri?: string;
  dbName?: string;
  collectionName?: string;
  openAIApiKey?: string;
  preferOpenAI?: boolean;
  mongoOptions?: MongoClientOptions; // Passed directly to MongoClient constructor
}
```

#### `Company`
```typescript
interface Company {
  company_name: string;
  website: string;
  short_description: string;
  product_description: string;
  mapped_function: string;
  mapped_industry: string;
  match_keywords: string;
  aliases: string;
  active: boolean | string;
  priority: string;
  source: string;
  zone: string;
  embedding?: number[];
  [key: string]: any;
}
```

---

## Development

To build and test the SDK locally:

```bash
# Clone the repository and install dependencies
cd sdk-js
npm install

# Run unit tests (runs MongoDB tests against mongodb://localhost:27017)
npm test

# Build files (CJS + ESM + DTS)
npm run build
```

---

## License

MIT © [horribleprogram](https://npmjs.com/~horribleprogram)

