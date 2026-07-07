import * as fs from 'fs';
import { MongoClient, Collection } from 'mongodb';
import { pipeline } from '@xenova/transformers';

export interface Company {
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

export interface MongoEmbeddingConfig {
  uri?: string;
  dbName?: string;
  collectionName?: string;
  openAIApiKey?: string;
  preferOpenAI?: boolean;
}

export function parseCSV(content: string): Record<string, string>[] {
  const lines: string[] = [];
  let currentLine = "";
  let insideQuote = false;
  
  for (let i = 0; i < content.length; i++) {
    const char = content[i];
    const nextChar = content[i + 1];
    
    if (char === '"') {
      insideQuote = !insideQuote;
    } else if (char === '\r' || char === '\n') {
      if (!insideQuote) {
        if (currentLine || i === 0 || content[i - 1] === '\n' || content[i - 1] === '\r') {
          lines.push(currentLine);
          currentLine = "";
        }
        if (char === '\r' && nextChar === '\n') {
          i++; // skip \n
        }
      } else {
        currentLine += char;
      }
    } else {
      currentLine += char;
    }
  }
  if (currentLine) {
    lines.push(currentLine);
  }
  
  if (lines.length === 0) return [];
  
  const parseLine = (line: string): string[] => {
    const result: string[] = [];
    let cell = "";
    let inside = false;
    for (let i = 0; i < line.length; i++) {
      const char = line[i];
      if (char === '"') {
        if (inside && line[i + 1] === '"') {
          cell += '"';
          i++;
        } else {
          inside = !inside;
        }
      } else if (char === ',' && !inside) {
        result.push(cell.trim());
        cell = "";
      } else {
        cell += char;
      }
    }
    result.push(cell.trim());
    return result;
  };
  
  const headers = parseLine(lines[0]);
  const records: Record<string, string>[] = [];
  
  for (let i = 1; i < lines.length; i++) {
    const cells = parseLine(lines[i]);
    if (cells.length === 1 && cells[0] === "") continue;
    
    const record: Record<string, string> = {};
    for (let j = 0; j < headers.length; j++) {
      record[headers[j]] = cells[j] || "";
    }
    records.push(record);
  }
  return records;
}

export function companyToText(company: Partial<Company>): string {
  const parts: string[] = [];
  const keys = [
    'company_name',
    'website',
    'short_description',
    'product_description',
    'mapped_function',
    'mapped_industry',
    'match_keywords',
    'aliases',
    'active',
    'priority',
    'source',
    'zone'
  ];
  for (const key of keys) {
    if (key in company) {
      const val = String(company[key]).trim();
      if (val) {
        parts.push(`${key}: ${val}`);
      }
    }
  }
  for (const [key, value] of Object.entries(company)) {
    if (keys.includes(key) || key === '_id' || key === 'embedding') continue;
    const val = String(value).trim();
    if (val) {
      parts.push(`${key}: ${val}`);
    }
  }
  return parts.join(' | ');
}

function cosineSimilarity(v1: number[], v2: number[]): number {
  let dotProduct = 0;
  let norm1 = 0;
  let norm2 = 0;
  for (let i = 0; i < v1.length; i++) {
    dotProduct += v1[i] * v2[i];
    norm1 += v1[i] * v1[i];
    norm2 += v2[i] * v2[i];
  }
  if (norm1 === 0 || norm2 === 0) return 0;
  return dotProduct / (Math.sqrt(norm1) * Math.sqrt(norm2));
}

export class CompanyEmbeddingManager {
  private client: MongoClient | null = null;
  private config: Required<MongoEmbeddingConfig>;
  private extractorPromise: any = null;

  constructor(config: MongoEmbeddingConfig = {}) {
    this.config = {
      uri: config.uri || process.env.MONGODB_URI || 'mongodb://localhost:27017',
      dbName: config.dbName || process.env.MONGODB_DB_NAME || 'distance_metrics',
      collectionName: config.collectionName || 'companies',
      openAIApiKey: config.openAIApiKey || process.env.OPENAI_API_KEY || '',
      preferOpenAI: config.preferOpenAI ?? !!(config.openAIApiKey || process.env.OPENAI_API_KEY),
    };
  }

  private async getExtractor() {
    if (!this.extractorPromise) {
      this.extractorPromise = (async () => {
        console.log("Loading local embedding model 'Xenova/all-MiniLM-L12-v2'...");
        const extractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L12-v2');
        console.log("Local embedding model loaded successfully.");
        return extractor;
      })();
    }
    return this.extractorPromise;
  }

  async connect(): Promise<MongoClient> {
    if (!this.client) {
      this.client = new MongoClient(this.config.uri);
      await this.client.connect();
    }
    return this.client;
  }

  async getCollection(): Promise<Collection<Company>> {
    const client = await this.connect();
    return client.db(this.config.dbName).collection<Company>(this.config.collectionName);
  }

  async disconnect(): Promise<void> {
    if (this.client) {
      await this.client.close();
      this.client = null;
    }
  }

  async getEmbedding(text: string): Promise<number[]> {
    if (this.config.preferOpenAI && this.config.openAIApiKey) {
      const embeddings = await this.getOpenAIBatchEmbeddings([text]);
      return embeddings[0];
    }

    const extractor = await this.getExtractor();
    const output = await extractor(text, { pooling: 'mean', normalize: true });
    return Array.from(output.data) as number[];
  }

  async getEmbeddings(texts: string[]): Promise<number[][]> {
    if (texts.length === 0) return [];
    
    if (this.config.preferOpenAI && this.config.openAIApiKey) {
      return this.getOpenAIBatchEmbeddings(texts);
    }

    const extractor = await this.getExtractor();
    const output = await extractor(texts, { pooling: 'mean', normalize: true });
    
    const dim = output.dims[1];
    const data = output.data;
    
    const embeddings: number[][] = [];
    for (let i = 0; i < texts.length; i++) {
      const startIndex = i * dim;
      const vector = Array.from(data.subarray(startIndex, startIndex + dim)) as number[];
      embeddings.push(vector);
    }
    return embeddings;
  }

  private async getOpenAIBatchEmbeddings(texts: string[]): Promise<number[][]> {
    const batchSize = 50;
    const results: number[][] = [];

    for (let i = 0; i < texts.length; i += batchSize) {
      const chunk = texts.slice(i, i + batchSize);
      const response = await fetch('https://api.openai.com/v1/embeddings', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${this.config.openAIApiKey}`,
        },
        body: JSON.stringify({
          input: chunk,
          model: 'text-embedding-3-small',
        }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`OpenAI API error: ${response.status} - ${errorText}`);
      }

      const json: any = await response.json();
      results.push(...json.data.map((item: any) => item.embedding));
    }

    return results;
  }

  async addCompany(company: Omit<Company, 'embedding'> & { embedding?: number[] }): Promise<void> {
    const col = await this.getCollection();
    
    let active = company.active;
    if (typeof active === 'string') {
      const activeStr = active.toLowerCase().trim();
      active = activeStr === 'true' || activeStr === 'yes' || activeStr === '1';
    }

    const cleanCompany: Company = {
      ...company,
      active: !!active,
    } as Company;

    if (!cleanCompany.embedding) {
      const text = companyToText(cleanCompany);
      cleanCompany.embedding = await this.getEmbedding(text);
    }

    await col.updateOne(
      { company_name: cleanCompany.company_name },
      { $set: cleanCompany },
      { upsert: true }
    );
  }

  async removeCompany(companyName: string): Promise<boolean> {
    const col = await this.getCollection();
    const result = await col.deleteOne({
      company_name: { $regex: new RegExp(`^${this.escapeRegExp(companyName)}$`, 'i') }
    });
    return (result.deletedCount ?? 0) > 0;
  }

  async queryCompanies(prompt: string, limit: number = 10): Promise<(Company & { similarity: number })[]> {
    const col = await this.getCollection();
    const queryVector = await this.getEmbedding(prompt);

    const allDocs = await col.find({}).toArray();
    
    const results = allDocs
      .filter(doc => doc.embedding && Array.isArray(doc.embedding))
      .map(doc => {
        const similarity = cosineSimilarity(queryVector, doc.embedding!);
        return {
          ...doc,
          similarity,
        };
      });

    results.sort((a, b) => b.similarity - a.similarity);

    return results.slice(0, limit);
  }

  async seedFromCSV(csvPath: string): Promise<number> {
    const csvContent = fs.readFileSync(csvPath, 'utf8');
    const records = parseCSV(csvContent);
    if (records.length === 0) return 0;

    const companies: Company[] = records.map(rec => {
      const activeStr = String(rec.active).toLowerCase().trim();
      const active = activeStr === 'true' || activeStr === 'yes' || activeStr === '1';
      return {
        company_name: rec.company_name || '',
        website: rec.website || '',
        short_description: rec.short_description || '',
        product_description: rec.product_description || '',
        mapped_function: rec.mapped_function || '',
        mapped_industry: rec.mapped_industry || '',
        match_keywords: rec.match_keywords || '',
        aliases: rec.aliases || '',
        active,
        priority: rec.priority || '',
        source: rec.source || '',
        zone: rec.zone || '',
      };
    }).filter(c => c.company_name);

    const texts = companies.map(c => companyToText(c));
    console.log(`Generating embeddings for ${companies.length} companies...`);
    const embeddings = await this.getEmbeddings(texts);

    for (let i = 0; i < companies.length; i++) {
      companies[i].embedding = embeddings[i];
    }

    const col = await this.getCollection();
    await col.deleteMany({});
    
    const batchSize = 100;
    for (let i = 0; i < companies.length; i += batchSize) {
      const batch = companies.slice(i, i + batchSize);
      await col.insertMany(batch);
    }

    return companies.length;
  }

  private escapeRegExp(string: string): string {
    return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  }
}

let defaultManager: CompanyEmbeddingManager | null = null;

function getDefaultManager(config?: MongoEmbeddingConfig): CompanyEmbeddingManager {
  if (!defaultManager || config) {
    defaultManager = new CompanyEmbeddingManager(config);
  }
  return defaultManager;
}

export async function seedCompanies(csvPath: string, config?: MongoEmbeddingConfig): Promise<number> {
  const manager = getDefaultManager(config);
  return manager.seedFromCSV(csvPath);
}

export async function addCompany(company: Omit<Company, 'embedding'> & { embedding?: number[] }, config?: MongoEmbeddingConfig): Promise<void> {
  const manager = getDefaultManager(config);
  return manager.addCompany(company);
}

export async function removeCompany(companyName: string, config?: MongoEmbeddingConfig): Promise<boolean> {
  const manager = getDefaultManager(config);
  return manager.removeCompany(companyName);
}

export async function queryCompanies(prompt: string, limit?: number, config?: MongoEmbeddingConfig): Promise<(Company & { similarity: number })[]> {
  const manager = getDefaultManager(config);
  return manager.queryCompanies(prompt, limit);
}

export async function closeConnection(): Promise<void> {
  if (defaultManager) {
    await defaultManager.disconnect();
    defaultManager = null;
  }
}
