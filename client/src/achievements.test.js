import {
  computeSharpEyeTrophy,
  computeRegularBirderTrophy,
  SHARP_EYE_TARGET,
  REGULAR_BIRDER_TARGET_DAYS,
} from './achievements';

function sighting({ confidence = 0.95, capturedAt = '2026-01-01T10:00:00.000Z' } = {}) {
  return { confidence, capturedAt };
}

describe('computeSharpEyeTrophy', () => {
  it('starts locked with no sightings', () => {
    const trophy = computeSharpEyeTrophy([]);
    expect(trophy.current).toBe(0);
    expect(trophy.unlocked).toBe(false);
  });

  it('only counts sightings at or above the confidence threshold (90%)', () => {
    const sightings = [
      sighting({ confidence: 0.95 }),
      sighting({ confidence: 0.9 }),
      sighting({ confidence: 0.89 }),
      sighting({ confidence: 0.5 }),
    ];
    const trophy = computeSharpEyeTrophy(sightings);
    expect(trophy.current).toBe(2);
    expect(trophy.qualifyingSightings).toHaveLength(2);
  });

  it('unlocks once qualifying sightings reach the target', () => {
    const sightings = Array.from({ length: SHARP_EYE_TARGET }, () => sighting({ confidence: 1 }));
    const trophy = computeSharpEyeTrophy(sightings);
    expect(trophy.current).toBe(SHARP_EYE_TARGET);
    expect(trophy.unlocked).toBe(true);
  });

  it('stays locked one short of the target', () => {
    const sightings = Array.from({ length: SHARP_EYE_TARGET - 1 }, () => sighting({ confidence: 1 }));
    const trophy = computeSharpEyeTrophy(sightings);
    expect(trophy.unlocked).toBe(false);
  });
});

describe('computeRegularBirderTrophy', () => {
  it('counts distinct calendar days, not distinct sightings', () => {
    const sightings = [
      sighting({ capturedAt: '2026-01-01T09:00:00.000Z' }),
      sighting({ capturedAt: '2026-01-01T18:00:00.000Z' }), // same day as above
      sighting({ capturedAt: '2026-01-02T09:00:00.000Z' }),
    ];
    const trophy = computeRegularBirderTrophy(sightings);
    expect(trophy.current).toBe(2);
  });

  it('unlocks once distinct days reach the target', () => {
    const sightings = Array.from({ length: REGULAR_BIRDER_TARGET_DAYS }, (_, i) =>
      sighting({ capturedAt: `2026-01-${String(i + 1).padStart(2, '0')}T09:00:00.000Z` })
    );
    const trophy = computeRegularBirderTrophy(sightings);
    expect(trophy.current).toBe(REGULAR_BIRDER_TARGET_DAYS);
    expect(trophy.unlocked).toBe(true);
  });

  it('has no qualifying-sightings list (nothing sensible to show for "distinct days")', () => {
    const trophy = computeRegularBirderTrophy([sighting()]);
    expect(trophy.qualifyingSightings).toBeNull();
  });
});
