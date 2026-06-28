import { describe, it, expect } from 'vitest';
import { Botcierge, query_intent } from '../src/index';

describe('Botcierge SDK (Local Mode)', () => {
  describe('Botcierge Class', () => {
    it('should correctly classify an utterance locally', async () => {
      const client = new Botcierge();
      const result = await client.query_intent('I am hungry');

      expect(result.domain).toBe('FOODLINK');
      expect(result.intent).toBe('request_food');
      expect(result.confidence).toBe('high');
      expect(result.scores).toBeDefined();
      expect(result.scores?.['request_food']).toBeGreaterThan(0.5);
    });
  });

  describe('query_intent helper', () => {
    it('should correctly classify an utterance using the helper', async () => {
      const result = await query_intent('I have food to share');

      expect(result.domain).toBe('FOODLINK');
      expect(result.intent).toBe('share_food');
      expect(result.confidence).toBe('high');
    });
  });
});


