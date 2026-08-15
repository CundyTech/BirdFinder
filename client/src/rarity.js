// Rarity tiers are relative to the app's own 60-species roster rather than
// fixed observation-count thresholds. Splitting by rank into 5 equal
// groups of 12 gives a clean partition — useful later for a "collect all
// of a tier" trophy system — and adapts automatically if the roster or
// counts change, instead of fixed thresholds drifting stale over time
// (e.g. leaving the "Very rare" tier permanently empty).
const TIER_DEFS = [
  { label: 'Very rare', pips: 5 },
  { label: 'Rare', pips: 4 },
  { label: 'Uncommon', pips: 3 },
  { label: 'Common', pips: 2 },
  { label: 'Very common', pips: 1 },
];

export const RARITY_MAX_PIPS = 5;
export const RARITY_TIER_COUNT = TIER_DEFS.length;

// rarityMap: { [speciesId]: ukObservationCount }.
// Returns { [speciesId]: { label, pips, rank } }, rank 0 = rarest overall.
export function computeRarityTiers(rarityMap) {
  if (!rarityMap) return null;

  const entries = Object.entries(rarityMap).filter(([, count]) => typeof count === 'number');
  if (entries.length === 0) return null;

  entries.sort((a, b) => a[1] - b[1]);

  const tierSize = entries.length / TIER_DEFS.length;
  const tiers = {};
  entries.forEach(([speciesId], rank) => {
    const tierIndex = Math.min(TIER_DEFS.length - 1, Math.floor(rank / tierSize));
    tiers[speciesId] = { ...TIER_DEFS[tierIndex], rank };
  });
  return tiers;
}
