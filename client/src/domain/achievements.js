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
