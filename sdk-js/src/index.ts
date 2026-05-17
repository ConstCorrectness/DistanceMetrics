export interface IntentResult {
  domain: string;
  intent: string;
  confidence: "high" | "medium" | "low";
  scores: Record<string, number>;
}

export interface BotciergeConfig {
  baseUrl?: string;
}

export class Botcierge {
  private baseUrl: string;

  constructor(config: BotciergeConfig = {}) {
    this.baseUrl = config.baseUrl || "http://localhost:8000";
  }

  async query_intent(utterance: string): Promise<IntentResult> {
    const response = await fetch(`${this.baseUrl}/classify`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ utterance }),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: "Unknown error" }));
      throw new Error(`Botcierge API error: ${error.detail || response.statusText}`);
    }

    return (await response.json()) as IntentResult;
  }
}

// Default instance for easy use
const defaultClient = new Botcierge();

/**
 * Classifies an utterance into an intent.
 * By default, connects to http://localhost:8000
 */
export async function query_intent(utterance: string): Promise<IntentResult> {
  return defaultClient.query_intent(utterance);
}
