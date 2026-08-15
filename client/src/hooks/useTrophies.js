import { useMemo } from 'react';
import { useSelector } from 'react-redux';
import { useGetUkRarityMapQuery } from '../services/birdInfoApi';
import { computeRarityTiers, RARITY_TIERS } from '../rarity';
import { SPECIES } from '../species';

// One trophy per rarity tier, earned by catching every species in that
// tier. Shared by the home screen (trophy count badge) and the trophy
// cabinet itself, so both read the exact same grouping/unlock logic.
// Returns null until the rarity data has loaded.
export default function useTrophies() {
  const { data: rarityMap } = useGetUkRarityMapQuery();
  const sightings = useSelector((state) => state.lifeList.sightings);

  return useMemo(() => {
    const tierBySpeciesId = computeRarityTiers(rarityMap);
    if (!tierBySpeciesId) return null;

    const caughtSpeciesIds = new Set(sightings.map((s) => s.speciesId));

    const speciesByTierLabel = new Map(RARITY_TIERS.map((t) => [t.label, []]));
    for (const species of SPECIES) {
      const tier = tierBySpeciesId[species.id];
      if (!tier) continue;
      speciesByTierLabel.get(tier.label)?.push(species);
    }

    return RARITY_TIERS.map((tierDef) => {
      const species = speciesByTierLabel.get(tierDef.label) || [];
      const caughtCount = species.filter((s) => caughtSpeciesIds.has(s.id)).length;
      return {
        label: tierDef.label,
        pips: tierDef.pips,
        species,
        caughtSpeciesIds,
        caughtCount,
        total: species.length,
        unlocked: species.length > 0 && caughtCount === species.length,
      };
    });
  }, [rarityMap, sightings]);
}
