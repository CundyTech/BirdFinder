import { computeRarityTiers, RARITY_TIERS, RARITY_MAX_PIPS } from './rarity';

describe('computeRarityTiers', () => {
  it('returns null when there is no rarity map yet', () => {
    expect(computeRarityTiers(null)).toBeNull();
    expect(computeRarityTiers(undefined)).toBeNull();
  });

  it('returns null when the map has no numeric counts', () => {
    expect(computeRarityTiers({})).toBeNull();
    expect(computeRarityTiers({ A: null, B: undefined })).toBeNull();
  });

  it('ignores non-numeric entries but still tiers the rest', () => {
    const result = computeRarityTiers({ A: 10, B: 'not a number', C: 20 });
    expect(result).not.toBeNull();
    expect(Object.keys(result)).toEqual(['A', 'C']);
  });

  it('splits an exactly-divisible roster into equal tiers, rarest first', () => {
    // 10 species, 5 tiers -> 2 per tier. Counts intentionally out of order
    // to prove sorting, not insertion order, decides the tier.
    const rarityMap = {
      J: 100, A: 1, I: 90, B: 2, H: 80, C: 3, G: 70, D: 4, F: 60, E: 5,
    };
    const result = computeRarityTiers(rarityMap);

    // Two rarest (lowest counts: A=1, B=2) land in the rarest tier.
    expect(result.A.label).toBe('Very rare');
    expect(result.B.label).toBe('Very rare');
    // Two most common (highest counts: J=100, I=90) land in the commonest tier.
    expect(result.J.label).toBe('Very common');
    expect(result.I.label).toBe('Very common');
  });

  it('assigns rank 0 to the single rarest entry', () => {
    const result = computeRarityTiers({ A: 5, B: 1, C: 3 });
    expect(result.B.rank).toBe(0);
  });

  it('every returned tier is one of the defined tiers with a valid pip count', () => {
    const rarityMap = Object.fromEntries(
      Array.from({ length: 20 }, (_, i) => [`S${i}`, i])
    );
    const result = computeRarityTiers(rarityMap);
    const labels = RARITY_TIERS.map((t) => t.label);

    for (const speciesId of Object.keys(rarityMap)) {
      const tier = result[speciesId];
      expect(labels).toContain(tier.label);
      expect(tier.pips).toBeGreaterThanOrEqual(1);
      expect(tier.pips).toBeLessThanOrEqual(RARITY_MAX_PIPS);
    }
  });
});
