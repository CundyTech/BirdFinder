import i18n from '../i18n';

// Rarity tiers are relative to the app's own 60-species roster rather than
// fixed observation-count thresholds. Splitting by rank into 5 equal
// groups of 12 gives a clean partition — useful later for a "collect all
// of a tier" trophy system — and adapts automatically if the roster or
// counts change, instead of fixed thresholds drifting stale over time
// (e.g. leaving the "Very rare" tier permanently empty).
//
// `key` is the frozen English label used as this tier's trophy-storage
// identity (see useTrophyCategories.js's makeTrophy) — it must never change
// even if `label`'s translation does, since it's part of the persisted
// claimed-trophy key.
const TIER_DEFS = [
  { key: 'Very rare', pips: 5 },
  { key: 'Rare', pips: 4 },
  { key: 'Uncommon', pips: 3 },
  { key: 'Common', pips: 2 },
  { key: 'Very common', pips: 1 },
];

export const RARITY_MAX_PIPS = 5;
export const RARITY_TIER_COUNT = TIER_DEFS.length;
// Rarest-first order, e.g. for iterating trophies or a tier legend.
export const RARITY_TIERS = TIER_DEFS.map((def) => ({ ...def, label: i18n.t(`rarityTiers.${def.key}`) }));

// rarityMap: { [speciesId]: ukObservationCount }.
// Returns { [speciesId]: { label, pips, rank } }, rank 0 = rarest overall.
export function computeRarityTiers(rarityMap) {
  if (!rarityMap) return null;

  const entries = Object.entries(rarityMap).filter(([, count]) => typeof count === 'number');
  if (entries.length === 0) return null;

  entries.sort((a, b) => a[1] - b[1]);

  const tierSize = entries.length / RARITY_TIERS.length;
  const tiers = {};
  entries.forEach(([speciesId], rank) => {
    const tierIndex = Math.min(RARITY_TIERS.length - 1, Math.floor(rank / tierSize));
    tiers[speciesId] = { ...RARITY_TIERS[tierIndex], rank };
  });
  return tiers;
}
