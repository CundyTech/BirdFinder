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
    label: 'Sharp Eye',
    description: `Save ${SHARP_EYE_TARGET} sightings with a strong-match identification (${LOW_CONFIDENCE_THRESHOLD}%+ confidence).`,
    unitLabel: 'strong matches',
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
    label: 'Regular Birder',
    description: `Log at least one sighting on ${REGULAR_BIRDER_TARGET_DAYS} different days.`,
    unitLabel: 'days',
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
  return [
    {
      type: 'behavior',
      label: 'First Reroll',
      description: `Save ${REROLL_MILESTONE_SIGHTINGS} sightings to earn a free reroll token (skips the Film cost once).`,
      unitLabel: 'sightings',
      current: Math.min(sightingsCount, REROLL_MILESTONE_SIGHTINGS),
      target: REROLL_MILESTONE_SIGHTINGS,
      unlocked: sightingsCount >= REROLL_MILESTONE_SIGHTINGS,
      qualifyingSightings: null,
      reward: { type: 'reroll', amount: 1 },
    },
    {
      type: 'behavior',
      label: 'Frequent Flyer',
      description: `Save ${FRAME_MILESTONE_SIGHTINGS} sightings to unlock the Floral Frame.`,
      unitLabel: 'sightings',
      current: Math.min(sightingsCount, FRAME_MILESTONE_SIGHTINGS),
      target: FRAME_MILESTONE_SIGHTINGS,
      unlocked: sightingsCount >= FRAME_MILESTONE_SIGHTINGS,
      qualifyingSightings: null,
      reward: { type: 'frame', frameId: 'floral-frame' },
    },
    {
      type: 'behavior',
      label: 'Century Club',
      description: `Save ${CENTURY_CLUB_SIGHTINGS} sightings to unlock the Polaroid Frame.`,
      unitLabel: 'sightings',
      current: Math.min(sightingsCount, CENTURY_CLUB_SIGHTINGS),
      target: CENTURY_CLUB_SIGHTINGS,
      unlocked: sightingsCount >= CENTURY_CLUB_SIGHTINGS,
      qualifyingSightings: null,
      reward: { type: 'frame', frameId: 'polaroid-frame' },
    },
    {
      type: 'behavior',
      label: 'Locked In',
      description: `Reach a ${STREAK_PROTECTION_MILESTONE_DAYS}-day streak to earn a streak protection token (covers one missed day).`,
      unitLabel: 'day streak',
      current: Math.min(longestStreak, STREAK_PROTECTION_MILESTONE_DAYS),
      target: STREAK_PROTECTION_MILESTONE_DAYS,
      unlocked: longestStreak >= STREAK_PROTECTION_MILESTONE_DAYS,
      qualifyingSightings: null,
      reward: { type: 'streakProtection', amount: 1 },
    },
    {
      type: 'behavior',
      label: 'Dedicated Birder',
      description: `Reach a ${STREAK_FRAME_MILESTONE_DAYS}-day streak to unlock the Wood Frame.`,
      unitLabel: 'day streak',
      current: Math.min(longestStreak, STREAK_FRAME_MILESTONE_DAYS),
      target: STREAK_FRAME_MILESTONE_DAYS,
      unlocked: longestStreak >= STREAK_FRAME_MILESTONE_DAYS,
      qualifyingSightings: null,
      reward: { type: 'frame', frameId: 'wood-frame' },
    },
    {
      type: 'behavior',
      label: 'Field Notes',
      description: `Log a sighting on ${FIELD_NOTES_DISTINCT_DAYS} different days to unlock the Calendar Frame.`,
      unitLabel: 'days',
      current: Math.min(distinctDays, FIELD_NOTES_DISTINCT_DAYS),
      target: FIELD_NOTES_DISTINCT_DAYS,
      unlocked: distinctDays >= FIELD_NOTES_DISTINCT_DAYS,
      qualifyingSightings: null,
      reward: { type: 'frame', frameId: 'calendar-frame' },
    },
    {
      type: 'behavior',
      label: "Completionist's Notes",
      description: 'Complete every Rarity trophy to unlock deep-dive lore on every species profile.',
      unitLabel: 'complete',
      current: rarityComplete ? 1 : 0,
      target: 1,
      unlocked: rarityComplete,
      qualifyingSightings: null,
      reward: { type: 'deepDiveLore' },
    },
    {
      type: 'behavior',
      label: 'Full Cabinet',
      description: 'Unlock every other trophy to earn the Baroque Frame.',
      unitLabel: 'complete',
      current: cabinetComplete ? 1 : 0,
      target: 1,
      unlocked: cabinetComplete,
      qualifyingSightings: null,
      reward: { type: 'frame', frameId: 'baroque-frame' },
    },
  ];
}
