import {
  computeSharpEyeTrophy,
  computeRegularBirderTrophy,
  computeMilestoneTrophies,
  SHARP_EYE_TARGET,
  REGULAR_BIRDER_TARGET_DAYS,
  REROLL_MILESTONE_SIGHTINGS,
  FRAME_MILESTONE_SIGHTINGS,
  CENTURY_CLUB_SIGHTINGS,
  STREAK_PROTECTION_MILESTONE_DAYS,
  STREAK_FRAME_MILESTONE_DAYS,
  FIELD_NOTES_DISTINCT_DAYS,
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

describe('computeMilestoneTrophies', () => {
  const baseCtx = {
    sightingsCount: 0,
    longestStreak: 0,
    distinctDays: 0,
    rarityComplete: false,
    cabinetComplete: false,
  };

  it('unlocks the reroll milestone once sightings reach the target and grants a reroll', () => {
    const [reroll] = computeMilestoneTrophies({ ...baseCtx, sightingsCount: REROLL_MILESTONE_SIGHTINGS });
    expect(reroll.unlocked).toBe(true);
    expect(reroll.reward).toEqual({ type: 'reroll', amount: 1 });
  });

  it('stays locked one sighting short of the reroll target', () => {
    const [reroll] = computeMilestoneTrophies({ ...baseCtx, sightingsCount: REROLL_MILESTONE_SIGHTINGS - 1 });
    expect(reroll.unlocked).toBe(false);
  });

  it('unlocks the frame milestone at the sightings target and grants the floral frame', () => {
    const trophies = computeMilestoneTrophies({ ...baseCtx, sightingsCount: FRAME_MILESTONE_SIGHTINGS });
    const frame = trophies.find((t) => t.label === 'Frequent Flyer');
    expect(frame.unlocked).toBe(true);
    expect(frame.reward).toEqual({ type: 'frame', frameId: 'floral-frame' });
  });

  it('unlocks the century-club milestone at a higher sightings target and grants the polaroid frame', () => {
    const trophies = computeMilestoneTrophies({ ...baseCtx, sightingsCount: CENTURY_CLUB_SIGHTINGS });
    const frame = trophies.find((t) => t.label === 'Century Club');
    expect(frame.unlocked).toBe(true);
    expect(frame.reward).toEqual({ type: 'frame', frameId: 'polaroid-frame' });
  });

  it('unlocks streak milestones off longestStreak, and grants protection then a frame', () => {
    const protection = computeMilestoneTrophies({ ...baseCtx, longestStreak: STREAK_PROTECTION_MILESTONE_DAYS }).find(
      (t) => t.label === 'Locked In'
    );
    expect(protection.unlocked).toBe(true);
    expect(protection.reward).toEqual({ type: 'streakProtection', amount: 1 });

    const woodFrame = computeMilestoneTrophies({ ...baseCtx, longestStreak: STREAK_FRAME_MILESTONE_DAYS }).find(
      (t) => t.label === 'Dedicated Birder'
    );
    expect(woodFrame.unlocked).toBe(true);
    expect(woodFrame.reward).toEqual({ type: 'frame', frameId: 'wood-frame' });
  });

  it('unlocks the field-notes milestone off distinctDays (not longestStreak) and grants the calendar frame', () => {
    const trophies = computeMilestoneTrophies({ ...baseCtx, distinctDays: FIELD_NOTES_DISTINCT_DAYS });
    const frame = trophies.find((t) => t.label === 'Field Notes');
    expect(frame.unlocked).toBe(true);
    expect(frame.reward).toEqual({ type: 'frame', frameId: 'calendar-frame' });
  });

  it('unlocks the lore milestone only once every rarity trophy is complete', () => {
    const locked = computeMilestoneTrophies({ ...baseCtx, rarityComplete: false }).find(
      (t) => t.label === "Completionist's Notes"
    );
    expect(locked.unlocked).toBe(false);

    const unlocked = computeMilestoneTrophies({ ...baseCtx, rarityComplete: true }).find(
      (t) => t.label === "Completionist's Notes"
    );
    expect(unlocked.unlocked).toBe(true);
    expect(unlocked.reward).toEqual({ type: 'deepDiveLore' });
  });

  it('unlocks the full-cabinet milestone only once the whole cabinet is complete, granting the baroque frame', () => {
    const trophy = computeMilestoneTrophies({ ...baseCtx, cabinetComplete: true }).find(
      (t) => t.label === 'Full Cabinet'
    );
    expect(trophy.unlocked).toBe(true);
    expect(trophy.reward).toEqual({ type: 'frame', frameId: 'baroque-frame' });
  });
});
