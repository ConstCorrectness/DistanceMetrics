import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import * as path from 'path';
import { 
  CompanyEmbeddingManager, 
  seedCompanies, 
  addCompany, 
  removeCompany, 
  queryCompanies, 
  closeConnection 
} from '../src/index';

const CSV_PATH = path.resolve(__dirname, '../../websummit_lisbon2025_companies.csv');
const MONGO_CONFIG = {
  uri: process.env.MONGODB_URI || 'mongodb://localhost:27017',
  dbName: 'distance_metrics_test',
  collectionName: 'companies'
};

describe('Company Embeddings & MongoDB Integration', () => {
  beforeAll(async () => {
    // Clean any prior run data
    const manager = new CompanyEmbeddingManager(MONGO_CONFIG);
    const col = await manager.getCollection();
    await col.deleteMany({});
    await manager.disconnect();
  });

  afterAll(async () => {
    await closeConnection();
  });

  it('should seed companies from CSV', async () => {
    console.log('Seeding companies from CSV...');
    const count = await seedCompanies(CSV_PATH, MONGO_CONFIG);
    console.log(`Seeded ${count} companies successfully.`);
    expect(count).toBeGreaterThan(400); // There are 438 companies
  }, 30000);

  it('should query top-k companies using semantic search', async () => {
    const results = await queryCompanies('AI customer support agents', 3, MONGO_CONFIG);
    console.log('Query: "AI customer support agents"');
    results.forEach((r, idx) => {
      console.log(`  ${idx + 1}. ${r.company_name} (Similarity: ${r.similarity.toFixed(4)}) - ${r.short_description}`);
    });

    expect(results.length).toBe(3);
    // Decagon AI or Parloa or similar should be near the top
    const names = results.map(r => r.company_name);
    expect(names.includes('Decagon AI') || names.includes('Intercom') || names.includes('Parloa')).toBe(true);
  });

  it('should add a custom company and retrieve it semantically', async () => {
    const customCompany = {
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
    };

    await addCompany(customCompany, MONGO_CONFIG);

    // Search for code assistant
    const results = await queryCompanies('autonomous coding assistant developed by Google DeepMind', 3, MONGO_CONFIG);
    expect(results.length).toBeGreaterThan(0);
    expect(results[0].company_name).toBe('Antigravity Code Labs');
    expect(results[0].similarity).toBeGreaterThan(0.5);
  });

  it('should remove a company', async () => {
    const removed = await removeCompany('Antigravity Code Labs', MONGO_CONFIG);
    expect(removed).toBe(true);

    // Confirm it is not returned in search results
    const results = await queryCompanies('autonomous coding assistant developed by Google DeepMind', 3, MONGO_CONFIG);
    const names = results.map(r => r.company_name);
    expect(names.includes('Antigravity Code Labs')).toBe(false);
  });
});
