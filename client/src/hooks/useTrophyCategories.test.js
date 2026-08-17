import { groupSpecies, makeTrophy } from './useTrophyCategories';
import { SPECIES } from '../domain/species';

describe('groupSpecies', () => {
  it('groups by id when group defs have one (TYPE_GROUPS/HABITAT_GROUPS/MIGRATION_GROUPS shape)', () => {
    const groupDefs = [{ id: 'even', label: 'Even' }, { id: 'odd', label: 'Odd' }];
    const idOf = new Map(SPECIES.map((s, i) => [s.id, i % 2 === 0 ? 'even' : 'odd']));

    const result = groupSpecies(groupDefs, (s) => idOf.get(s.id));

    expect(result.get('even').length + result.get('odd').length).toBe(SPECIES.length);
    expect(result.get('even').every((s) => idOf.get(s.id) === 'even')).toBe(true);
  });

  // Regression test: RARITY_TIERS entries only have {label, pips}, no id.
  // groupSpecies used to key its lookup map by group.id unconditionally,
  // so every rarity tier's key came out `undefined` and every species
  // silently vanished into a bucket nothing ever read from — every rarity
  // trophy showed 0/0 instead of its real count.
  it('falls back to label as the key when group defs have no id (RARITY_TIERS shape)', () => {
    const groupDefs = [{ label: 'Alpha' }, { label: 'Beta' }];
    const labelOf = new Map(SPECIES.map((s, i) => [s.id, i % 2 === 0 ? 'Alpha' : 'Beta']));

    const result = groupSpecies(groupDefs, (s) => labelOf.get(s.id));

    expect(result.has(undefined)).toBe(false);
    expect(result.get('Alpha').length).toBeGreaterThan(0);
    expect(result.get('Beta').length).toBeGreaterThan(0);
    expect(result.get('Alpha').length + result.get('Beta').length).toBe(SPECIES.length);
  });

  it('skips species whose getGroupIdForSpecies returns nothing', () => {
    const groupDefs = [{ id: 'only' }];
    const result = groupSpecies(groupDefs, () => null);
    expect(result.get('only')).toHaveLength(0);
  });
});

describe('makeTrophy', () => {
  it('counts how many of the group are caught', () => {
    const species = SPECIES.slice(0, 3);
    const caughtSpeciesIds = new Set([species[0].id, species[1].id]);

    const trophy = makeTrophy('Test', species, caughtSpeciesIds);

    expect(trophy.total).toBe(3);
    expect(trophy.caughtCount).toBe(2);
    expect(trophy.unlocked).toBe(false);
  });

  it('unlocks only when every species in the group is caught', () => {
    const species = SPECIES.slice(0, 2);
    const caughtSpeciesIds = new Set(species.map((s) => s.id));

    const trophy = makeTrophy('Test', species, caughtSpeciesIds);

    expect(trophy.unlocked).toBe(true);
  });

  it('never unlocks an empty group', () => {
    const trophy = makeTrophy('Empty', [], new Set());
    expect(trophy.total).toBe(0);
    expect(trophy.unlocked).toBe(false);
  });
});
