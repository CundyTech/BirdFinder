import { useMemo } from 'react';
import { useSelector } from 'react-redux';
import i18n from '../i18n';
import { useGetUkRarityMapQuery } from '../services/birdInfoApi';
import { computeRarityTiers, RARITY_TIERS } from '../domain/rarity';
import { SPECIES, TYPE_GROUPS, HABITAT_GROUPS, MIGRATION_GROUPS } from '../domain/species';
import { computeSharpEyeTrophy, computeRegularBirderTrophy, computeMilestoneTrophies } from '../domain/achievements';
import { DEBUG_UNLOCK_EVERYTHING } from '../config';

// DEBUG_UNLOCK_EVERYTHING support: forces a single trophy to its fully-
// unlocked state regardless of what actually happened, for both trophy
// shapes ('species' — group of birds; 'behavior' — a progress counter).
// Applied as a final pass below so nothing upstream needs to know about it.
export function forceUnlocked(trophy) {
  if (trophy.type === 'species') {
    return {
      ...trophy,
      caughtSpeciesIds: new Set(trophy.species.map((s) => s.id)),
      caughtCount: trophy.total,
      unlocked: trophy.total > 0,
    };
  }
  return { ...trophy, current: trophy.target, unlocked: true };
}

// groupDefs is either TYPE_GROUPS/HABITAT_GROUPS/MIGRATION_GROUPS ({id, key,
// label}) or RARITY_TIERS ({key, label, pips} — no id), so keys fall back to
// `key` (the frozen, untranslated identity — never `label`, which is
// translated and so unsafe to use as a lookup/grouping key); getGroupIdForSpecies
// must return the matching one in each case.
export function groupSpecies(groupDefs, getGroupIdForSpecies) {
  const speciesByGroupId = new Map(groupDefs.map((g) => [g.id ?? g.key, []]));
  for (const species of SPECIES) {
    const groupId = getGroupIdForSpecies(species);
    if (groupId) speciesByGroupId.get(groupId)?.push(species);
  }
  return speciesByGroupId;
}

// `key` is the frozen English identity used as this trophy's persisted
// claim-key (see App.js, PerksModal.js) — must never change even if `label`
// is displayed in a different language. `label` is the translated text.
export function makeTrophy(key, label, species, caughtSpeciesIds) {
  const caughtCount = species.filter((s) => caughtSpeciesIds.has(s.id)).length;
  return {
    type: 'species',
    key,
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
  const longestStreak = useSelector((state) => state.streak.longestStreak);

  return useMemo(() => {
    const caughtSpeciesIds = new Set(sightings.map((s) => s.speciesId));

    const tierBySpeciesId = computeRarityTiers(rarityMap);
    let rarityTrophies = null;
    if (tierBySpeciesId) {
      const speciesByTier = groupSpecies(RARITY_TIERS, (s) => tierBySpeciesId[s.id]?.key);
      rarityTrophies = RARITY_TIERS.map((def) =>
        makeTrophy(def.key, def.label, speciesByTier.get(def.key) || [], caughtSpeciesIds)
      );
    }

    const speciesByType = groupSpecies(TYPE_GROUPS, (s) => s.typeGroup);
    const typeTrophies = TYPE_GROUPS.map((def) =>
      makeTrophy(def.key, def.label, speciesByType.get(def.id) || [], caughtSpeciesIds)
    );

    const speciesByHabitat = groupSpecies(HABITAT_GROUPS, (s) => s.habitat);
    const habitatTrophies = HABITAT_GROUPS.map((def) =>
      makeTrophy(def.key, def.label, speciesByHabitat.get(def.id) || [], caughtSpeciesIds)
    );

    const speciesByMigration = groupSpecies(MIGRATION_GROUPS, (s) => s.migration);
    const migrationTrophies = MIGRATION_GROUPS.map((def) =>
      makeTrophy(def.key, def.label, speciesByMigration.get(def.id) || [], caughtSpeciesIds)
    );

    const regularBirderTrophy = computeRegularBirderTrophy(sightings);
    const achievementTrophies = [
      makeTrophy('Full Flock', i18n.t('achievements.fullFlock'), SPECIES, caughtSpeciesIds),
      computeSharpEyeTrophy(sightings),
      regularBirderTrophy,
    ];

    // rarityTrophies is null while its rarity-map fetch is still loading —
    // treat that as "not complete yet" rather than blocking.
    const rarityComplete = Boolean(rarityTrophies?.length) && rarityTrophies.every((t) => t.unlocked);
    const cabinetComplete =
      rarityComplete &&
      [typeTrophies, habitatTrophies, migrationTrophies, achievementTrophies].every((list) =>
        list.every((t) => t.unlocked)
      );

    const milestoneTrophies = computeMilestoneTrophies({
      sightingsCount: sightings.length,
      longestStreak,
      distinctDays: regularBirderTrophy.current,
      rarityComplete,
      cabinetComplete,
    });

    const categories = [
      {
        id: 'rarity',
        label: i18n.t('trophyCategories.rarity.label'),
        description: i18n.t('trophyCategories.rarity.description'),
        trophies: rarityTrophies,
      },
      {
        id: 'types',
        label: i18n.t('trophyCategories.types.label'),
        description: i18n.t('trophyCategories.types.description'),
        trophies: typeTrophies,
      },
      {
        id: 'habitats',
        label: i18n.t('trophyCategories.habitats.label'),
        description: i18n.t('trophyCategories.habitats.description'),
        trophies: habitatTrophies,
      },
      {
        id: 'migration',
        label: i18n.t('trophyCategories.migration.label'),
        description: i18n.t('trophyCategories.migration.description'),
        trophies: migrationTrophies,
      },
      {
        id: 'achievements',
        label: i18n.t('trophyCategories.achievements.label'),
        description: i18n.t('trophyCategories.achievements.description'),
        trophies: achievementTrophies,
      },
      {
        id: 'milestones',
        label: i18n.t('trophyCategories.milestones.label'),
        description: i18n.t('trophyCategories.milestones.description'),
        trophies: milestoneTrophies,
      },
    ];

    if (!DEBUG_UNLOCK_EVERYTHING) return categories;
    return categories.map((category) => ({
      ...category,
      trophies: category.trophies ? category.trophies.map(forceUnlocked) : category.trophies,
    }));
  }, [rarityMap, sightings, longestStreak]);
}
