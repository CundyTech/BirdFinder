import { useMemo } from 'react';
import { useSelector } from 'react-redux';
import { useGetUkRarityMapQuery } from '../services/birdInfoApi';
import { computeRarityTiers, RARITY_TIERS } from '../rarity';
import { SPECIES, TYPE_GROUPS, HABITAT_GROUPS, MIGRATION_GROUPS } from '../species';
import { computeSharpEyeTrophy, computeRegularBirderTrophy } from '../achievements';

// groupDefs is either TYPE_GROUPS/HABITAT_GROUPS/MIGRATION_GROUPS ({id,
// label}) or RARITY_TIERS ({label, pips} — no id), so keys fall back to
// label; getGroupIdForSpecies must return the matching one in each case.
export function groupSpecies(groupDefs, getGroupIdForSpecies) {
  const speciesByGroupId = new Map(groupDefs.map((g) => [g.id ?? g.label, []]));
  for (const species of SPECIES) {
    const groupId = getGroupIdForSpecies(species);
    if (groupId) speciesByGroupId.get(groupId)?.push(species);
  }
  return speciesByGroupId;
}

export function makeTrophy(label, species, caughtSpeciesIds) {
  const caughtCount = species.filter((s) => caughtSpeciesIds.has(s.id)).length;
  return {
    type: 'species',
    label,
    species,
    caughtSpeciesIds,
    caughtCount,
    total: species.length,
    unlocked: species.length > 0 && caughtCount === species.length,
  };
}

// Every trophy belongs to one of five top-level categories:
//  - Rarity: tier-based (same ranking as useSpeciesRarity)
//  - Bird Families / Habitats / Migration: informal groupings in species.js
//  - Achievements: not species-grouping at all — a completionist trophy
//    plus behavior-based ones computed straight from the sightings log
// A category's `trophies` is null while its underlying data (only Rarity
// needs a network fetch) is still loading, so the screen can show a
// per-category loading state instead of blocking on everything at once.
export default function useTrophyCategories() {
  const { data: rarityMap } = useGetUkRarityMapQuery();
  const sightings = useSelector((state) => state.lifeList.sightings);

  return useMemo(() => {
    const caughtSpeciesIds = new Set(sightings.map((s) => s.speciesId));

    const tierBySpeciesId = computeRarityTiers(rarityMap);
    let rarityTrophies = null;
    if (tierBySpeciesId) {
      const speciesByTier = groupSpecies(RARITY_TIERS, (s) => tierBySpeciesId[s.id]?.label);
      rarityTrophies = RARITY_TIERS.map((def) =>
        makeTrophy(def.label, speciesByTier.get(def.label) || [], caughtSpeciesIds)
      );
    }

    const speciesByType = groupSpecies(TYPE_GROUPS, (s) => s.typeGroup);
    const typeTrophies = TYPE_GROUPS.map((def) =>
      makeTrophy(def.label, speciesByType.get(def.id) || [], caughtSpeciesIds)
    );

    const speciesByHabitat = groupSpecies(HABITAT_GROUPS, (s) => s.habitat);
    const habitatTrophies = HABITAT_GROUPS.map((def) =>
      makeTrophy(def.label, speciesByHabitat.get(def.id) || [], caughtSpeciesIds)
    );

    const speciesByMigration = groupSpecies(MIGRATION_GROUPS, (s) => s.migration);
    const migrationTrophies = MIGRATION_GROUPS.map((def) =>
      makeTrophy(def.label, speciesByMigration.get(def.id) || [], caughtSpeciesIds)
    );

    const achievementTrophies = [
      makeTrophy('Full Flock', SPECIES, caughtSpeciesIds),
      computeSharpEyeTrophy(sightings),
      computeRegularBirderTrophy(sightings),
    ];

    return [
      {
        id: 'rarity',
        label: 'Rarity',
        description: 'Catch every species in a rarity tier.',
        trophies: rarityTrophies,
      },
      {
        id: 'types',
        label: 'Bird Families',
        description: 'Catch every species in a family or type.',
        trophies: typeTrophies,
      },
      {
        id: 'habitats',
        label: 'Habitats',
        description: 'Catch every species typically found in a habitat.',
        trophies: habitatTrophies,
      },
      {
        id: 'migration',
        label: 'Migration',
        description: 'Catch every species with a given migratory pattern.',
        trophies: migrationTrophies,
      },
      {
        id: 'achievements',
        label: 'Achievements',
        description: 'Milestones for how you use the app, not just what you catch.',
        trophies: achievementTrophies,
      },
    ];
  }, [rarityMap, sightings]);
}
