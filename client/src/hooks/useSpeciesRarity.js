import { useMemo } from 'react';
import { useGetUkRarityMapQuery } from '../services/birdInfoApi';
import { computeRarityTiers } from '../rarity';

// Rarity is ranked relative to the whole 60-species roster, so every
// consumer needs the full counts map to rank against — this hook does
// that ranking once (memoized on the shared, cached query result) and
// hands back just the one species' tier.
export default function useSpeciesRarity(speciesId) {
  const { data: rarityMap } = useGetUkRarityMapQuery();
  const tiers = useMemo(() => computeRarityTiers(rarityMap), [rarityMap]);
  return tiers?.[speciesId] ?? null;
}
