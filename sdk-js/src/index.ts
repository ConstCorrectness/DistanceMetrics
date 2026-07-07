import { pipeline } from '@xenova/transformers';
import { INTENTS_TAXONOMY } from './intents';

export interface IntentResult {
  domain: string;
  intent: string;
  confidence: "high" | "medium" | "low";
  scores?: Record<string, number>;
}

export interface BotciergeConfig {}

let extractorPromise: any = null;
function getExtractor() {
  if (!extractorPromise) {
    extractorPromise = (async () => {
      console.log("Loading local embedding model 'Xenova/all-MiniLM-L12-v2'...");
      console.log("Note: On first run, this will download the model files (approx. 120MB).");
      const extractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L12-v2');
      console.log("Local embedding model loaded successfully.");
      return extractor;
    })();
  }
  return extractorPromise;
}

interface EmbeddedTaxonomyItem {
  domain: string;
  intent: string;
  utterance: string;
  vector: number[];
}

let cachedTaxonomyEmbeddingsPromise: Promise<EmbeddedTaxonomyItem[]> | null = null;

async function getTaxonomyEmbeddings(): Promise<EmbeddedTaxonomyItem[]> {
  if (cachedTaxonomyEmbeddingsPromise) {
    return cachedTaxonomyEmbeddingsPromise;
  }

  cachedTaxonomyEmbeddingsPromise = (async () => {
    const extractor = await getExtractor();
    const items: { domain: string; intent: string; utterance: string }[] = [];
    
    for (const [domain, intents] of Object.entries(INTENTS_TAXONOMY)) {
      for (const [intent, utterances] of Object.entries(intents)) {
        for (const utterance of utterances) {
          if (utterance.trim()) {
            items.push({ domain, intent, utterance });
          }
        }
      }
    }

    const texts = items.map(item => item.utterance);
    const output = await extractor(texts, { pooling: 'mean', normalize: true });
    
    const dim = output.dims[1];
    const data = output.data;
    
    const embeddings: EmbeddedTaxonomyItem[] = [];
    for (let i = 0; i < items.length; i++) {
      const startIndex = i * dim;
      const vector = Array.from(data.subarray(startIndex, startIndex + dim)) as number[];
      embeddings.push({
        ...items[i],
        vector,
      });
    }
    
    return embeddings;
  })();

  return cachedTaxonomyEmbeddingsPromise;
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

export class Botcierge {
  constructor(config: BotciergeConfig = {}) {}

  async query_intent(utterance: string): Promise<IntentResult> {
    // Local classification using Xenova/all-MiniLM-L12-v2
    const extractor = await getExtractor();
    const output = await extractor(utterance, { pooling: 'mean', normalize: true });
    const queryVector = Array.from(output.data) as number[];
    
    const taxEmbeddings = await getTaxonomyEmbeddings();
    
    let bestSim = -1;
    let bestItem: EmbeddedTaxonomyItem | null = null;
    
    for (const item of taxEmbeddings) {
      const sim = cosineSimilarity(queryVector, item.vector);
      if (sim > bestSim) {
        bestSim = sim;
        bestItem = item;
      }
    }
    
    if (!bestItem || bestSim < 0.35) {
      return {
        domain: "unknown",
        intent: "unknown",
        confidence: "low",
        scores: {},
      };
    }
    
    let confidence: "high" | "medium" | "low" = "low";
    if (bestSim >= 0.70) {
      confidence = "high";
    } else if (bestSim >= 0.50) {
      confidence = "medium";
    }
    
    return {
      domain: bestItem.domain,
      intent: bestItem.intent,
      confidence,
      scores: {
        [bestItem.intent]: bestSim,
      },
    };
  }
}

// Default instance for easy use
const defaultClient = new Botcierge();

/**
 * Classifies an utterance into an intent.
 */
export async function query_intent(utterance: string): Promise<IntentResult> {
  return defaultClient.query_intent(utterance);
}

export * from './companyEmbeddings';


