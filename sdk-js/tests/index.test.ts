import { describe, it, expect, vi, beforeEach } from 'vitest';
import { Botcierge, query_intent } from '../src/index';

// Mock global fetch
global.fetch = vi.fn();

describe('Botcierge SDK', () => {
  beforeEach(() => {
    vi.resetAllMocks();
  });

  describe('Botcierge Class', () => {
    it('should use the provided baseUrl', async () => {
      const customUrl = 'https://api.example.com';
      const client = new Botcierge({ baseUrl: customUrl });
      
      const mockResult = {
        domain: 'test-domain',
        intent: 'test-intent',
        confidence: 'high',
        scores: { 'test-intent': 0.9 }
      };

      (fetch as any).mockResolvedValue({
        ok: true,
        json: async () => mockResult,
      });

      await client.query_intent('hello');

      expect(fetch).toHaveBeenCalledWith(
        expect.stringContaining(customUrl),
        expect.any(Object)
      );
    });

    it('should throw an error if the API response is not ok', async () => {
      const client = new Botcierge();
      
      (fetch as any).mockResolvedValue({
        ok: false,
        statusText: 'Forbidden',
        json: async () => ({ detail: 'Invalid token' }),
      });

      await expect(client.query_intent('hello')).rejects.toThrow('Botcierge API error: Invalid token');
    });
  });

  describe('query_intent helper', () => {
    it('should correctly classify an utterance using the default client', async () => {
      const mockResult = {
        domain: 'moneyshare',
        intent: 'request_loan',
        confidence: 'high',
        scores: { 'request_loan': 0.95 }
      };

      (fetch as any).mockResolvedValue({
        ok: true,
        json: async () => mockResult,
      });

      const result = await query_intent('can I borrow some money');

      expect(result).toEqual(mockResult);
      expect(fetch).toHaveBeenCalledWith(
        'http://localhost:8000/classify',
        expect.objectContaining({
          method: 'POST',
          body: JSON.stringify({ utterance: 'can I borrow some money' })
        })
      );
    });
  });
});
