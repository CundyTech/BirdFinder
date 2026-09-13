import i18n from '../i18n';
import { LOW_CONFIDENCE_THRESHOLD } from '../config';

// Behavior-based trophies — about how you use the app (accuracy, regular
// use) rather than which species you've caught, so they're computed
// straight from the sightings log instead of grouping the species roster.
// Targets are deliberately modest: these should reward using the app
// normally, not require grinding.
export const SHARP_EYE_TARGET = 10;
export const REGULAR_BIRDER_TARGET_DAYS = 7;

export function computeSharpEyeTrophy(sightings) {
  const qualifying = sightings.filter((s) => (s.confidence || 0) * 100 >= LOW_CONFIDENCE_THRESHOLD);
  return {
    type: 'behavior',
    // Frozen English identity used as this trophy's persisted claim-key
    // (see useTrophyCategories.js's makeTrophy and App.js) — must never
    // change even if `label`'s translation does.
    key: 'Sharp Eye',
    label: i18n.t('achievements.sharpEye.label'),
    description: i18n.t('achievements.sharpEye.description', {
      target: SHARP_EYE_TARGET,
      threshold: LOW_CONFIDENCE_THRESHOLD,
    }),
    unitLabel: i18n.t('achievements.sharpEye.unitLabel'),
    current: qualifying.length,
    target: SHARP_EYE_TARGET,
    unlocked: qualifying.length >= SHARP_EYE_TARGET,
    qualifyingSightings: qualifying,
  };
}

export function computeRegularBirderTrophy(sightings) {
  // Calendar day in UTC — a simplification, but exact local-timezone
  // midnight boundaries aren't worth the complexity for a trophy about
  // "did you come back another day."
  const distinctDays = new Set(sightings.map((s) => s.capturedAt.slice(0, 10)));
  return {
    type: 'behavior',
    key: 'Regular Birder',
    label: i18n.t('achievements.regularBirder.label'),
    description: i18n.t('achievements.regularBirder.description', { target: REGULAR_BIRDER_TARGET_DAYS }),
    unitLabel: i18n.t('achievements.regularBirder.unitLabel'),
    current: distinctDays.size,
    target: REGULAR_BIRDER_TARGET_DAYS,
    unlocked: distinctDays.size >= REGULAR_BIRDER_TARGET_DAYS,
    qualifyingSightings: null,
  };
}

// Milestone trophies grant a one-off non-Film reward (see rewardsSlice.js)
// instead of the usual Film payout every other trophy gives — that
// substitution happens in App.js's claim effect, keyed off category id
// 'milestones', not here.
export const REROLL_MILESTONE_SIGHTINGS = 10;
export const FRAME_MILESTONE_SIGHTINGS = 25;
export const CENTURY_CLUB_SIGHTINGS = 50;
export const STREAK_PROTECTION_MILESTONE_DAYS = 7;
export const STREAK_FRAME_MILESTONE_DAYS = 30;
export const FIELD_NOTES_DISTINCT_DAYS = 30;

export function computeMilestoneTrophies({
  sightingsCount,
  longestStreak,
  distinctDays,
  rarityComplete,
  cabinetComplete,
}) {
  // Each `key` is the frozen English identity used as this milestone's
  // persisted claim-key (see PerksModal.js, App.js) — must never change
  // even if `label`'s translation does.
  return [
    {
      type: 'behavior',
      key: 'First Reroll',
      label: i18n.t('achievements.milestones.firstReroll.label'),
      description: i18n.t('achievements.milestones.firstReroll.description', { target: REROLL_MILESTONE_SIGHTINGS }),
      unitLabel: i18n.t('achievements.milestones.firstReroll.unitLabel'),
      current: Math.min(sightingsCount, REROLL_MILESTONE_SIGHTINGS),
      target: REROLL_MILESTONE_SIGHTINGS,
      unlocked: sightingsCount >= REROLL_MILESTONE_SIGHTINGS,
      qualifyingSightings: null,
      reward: { type: 'reroll', amount: 1 },
    },
    {
      type: 'behavior',
      key: 'Frequent Flyer',
      label: i18n.t('achievements.milestones.frequentFlyer.label'),
      description: i18n.t('achievements.milestones.frequentFlyer.description', { target: FRAME_MILESTONE_SIGHTINGS }),
      unitLabel: i18n.t('achievements.milestones.frequentFlyer.unitLabel'),
      current: Math.min(sightingsCount, FRAME_MILESTONE_SIGHTINGS),
      target: FRAME_MILESTONE_SIGHTINGS,
      unlocked: sightingsCount >= FRAME_MILESTONE_SIGHTINGS,
      qualifyingSightings: null,
      reward: { type: 'frame', frameId: 'floral-frame' },
    },
    {
      type: 'behavior',
      key: 'Century Club',
      label: i18n.t('achievements.milestones.centuryClub.label'),
      description: i18n.t('achievements.milestones.centuryClub.description', { target: CENTURY_CLUB_SIGHTINGS }),
      unitLabel: i18n.t('achievements.milestones.centuryClub.unitLabel'),
      current: Math.min(sightingsCount, CENTURY_CLUB_SIGHTINGS),
      target: CENTURY_CLUB_SIGHTINGS,
      unlocked: sightingsCount >= CENTURY_CLUB_SIGHTINGS,
      qualifyingSightings: null,
      reward: { type: 'frame', frameId: 'polaroid-frame' },
    },
    {
      type: 'behavior',
      key: 'Locked In',
      label: i18n.t('achievements.milestones.lockedIn.label'),
      description: i18n.t('achievements.milestones.lockedIn.description', { target: STREAK_PROTECTION_MILESTONE_DAYS }),
      unitLabel: i18n.t('achievements.milestones.lockedIn.unitLabel'),
      current: Math.min(longestStreak, STREAK_PROTECTION_MILESTONE_DAYS),
      target: STREAK_PROTECTION_MILESTONE_DAYS,
      unlocked: longestStreak >= STREAK_PROTECTION_MILESTONE_DAYS,
      qualifyingSightings: null,
      reward: { type: 'streakProtection', amount: 1 },
    },
    {
      type: 'behavior',
      key: 'Dedicated Birder',
      label: i18n.t('achievements.milestones.dedicatedBirder.label'),
      description: i18n.t('achievements.milestones.dedicatedBirder.description', { target: STREAK_FRAME_MILESTONE_DAYS }),
      unitLabel: i18n.t('achievements.milestones.dedicatedBirder.unitLabel'),
      current: Math.min(longestStreak, STREAK_FRAME_MILESTONE_DAYS),
      target: STREAK_FRAME_MILESTONE_DAYS,
      unlocked: longestStreak >= STREAK_FRAME_MILESTONE_DAYS,
      qualifyingSightings: null,
      reward: { type: 'frame', frameId: 'wood-frame' },
    },
    {
      type: 'behavior',
      key: 'Field Notes',
      label: i18n.t('achievements.milestones.fieldNotes.label'),
      description: i18n.t('achievements.milestones.fieldNotes.description', { target: FIELD_NOTES_DISTINCT_DAYS }),
      unitLabel: i18n.t('achievements.milestones.fieldNotes.unitLabel'),
      current: Math.min(distinctDays, FIELD_NOTES_DISTINCT_DAYS),
      target: FIELD_NOTES_DISTINCT_DAYS,
      unlocked: distinctDays >= FIELD_NOTES_DISTINCT_DAYS,
      qualifyingSightings: null,
      reward: { type: 'frame', frameId: 'calendar-frame' },
    },
    {
      type: 'behavior',
      key: "Completionist's Notes",
      label: i18n.t("achievements.milestones.completionistsNotes.label"),
      description: i18n.t('achievements.milestones.completionistsNotes.description'),
      unitLabel: i18n.t('achievements.milestones.completionistsNotes.unitLabel'),
      current: rarityComplete ? 1 : 0,
      target: 1,
      unlocked: rarityComplete,
      qualifyingSightings: null,
      reward: { type: 'deepDiveLore' },
    },
    {
      type: 'behavior',
      key: 'Full Cabinet',
      label: i18n.t('achievements.milestones.fullCabinet.label'),
      description: i18n.t('achievements.milestones.fullCabinet.description'),
      unitLabel: i18n.t('achievements.milestones.fullCabinet.unitLabel'),
      current: cabinetComplete ? 1 : 0,
      target: 1,
      unlocked: cabinetComplete,
      qualifyingSightings: null,
      reward: { type: 'frame', frameId: 'baroque-frame' },
    },
  ];
}
