import i18n from '../i18n';
import { SPECIES_IDS } from './species';

// Curated "fun profile" content for the life-list facts card: general
// ornithological knowledge for common, well-documented UK species, not
// pulled from a live API like the rarity/taxonomy data elsewhere in this
// app (see species.js). The actual text lives in i18n/locales/en/*.json —
// worth a sanity check there if anything reads oddly for a specific species.
export const BIRD_PROFILES = Object.fromEntries(
  SPECIES_IDS.filter((id) => i18n.exists(`birdProfiles.${id}`)).map((id) => [
    id,
    {
      fact: i18n.t(`birdProfiles.${id}.fact`),
      prey: i18n.t(`birdProfiles.${id}.prey`),
      predators: i18n.t(`birdProfiles.${id}.predators`),
    },
  ])
);

// A second, distinct fact per species — unlocked as a milestone reward
// rather than shown by default (see rewardsSlice.js's deepDiveLoreUnlocked).
// Deliberately a different angle from BIRD_PROFILES' `fact` field (name
// origin, conservation history, etc.) rather than a repeat of it.
export const DEEP_DIVE = Object.fromEntries(
  SPECIES_IDS.filter((id) => i18n.exists(`deepDive.${id}`)).map((id) => [id, i18n.t(`deepDive.${id}`)])
);

// Only species that genuinely leave Britain for winter (see species.js's
// MIGRATION_GROUPS) have a route. Everyone else is a year-round resident,
// shown as plain text instead of an empty/misleading map.
export const MIGRATION_ROUTES = Object.fromEntries(
  SPECIES_IDS.filter((id) => i18n.exists(`migrationRoutes.${id}`)).map((id) => [
    id,
    {
      breeding: i18n.t(`migrationRoutes.${id}.breeding`),
      wintering: i18n.t(`migrationRoutes.${id}.wintering`),
    },
  ])
);
