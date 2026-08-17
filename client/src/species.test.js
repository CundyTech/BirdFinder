import {
  SPECIES,
  SPECIES_COUNT,
  TYPE_GROUPS,
  HABITAT_GROUPS,
  MIGRATION_GROUPS,
  TAXON_IDS,
  formatSpeciesName,
} from './species';

// This roster is hand-curated (taxon ids verified against the live
// iNaturalist API, groupings assigned by hand) — these tests exist because
// we've already shipped bugs from it going out of sync (a grouping id
// mismatch that silently dropped every rarity trophy to 0/0).
describe('species roster integrity', () => {
  it('has 60 species', () => {
    expect(SPECIES_COUNT).toBe(60);
    expect(SPECIES).toHaveLength(60);
  });

  it('every species has a unique id', () => {
    const ids = SPECIES.map((s) => s.id);
    expect(new Set(ids).size).toBe(ids.length);
  });

  it('every species has a numeric taxonId', () => {
    for (const species of SPECIES) {
      expect(typeof species.taxonId).toBe('number');
      expect(Number.isFinite(species.taxonId)).toBe(true);
    }
  });

  it('every species has a taxonId entry in TAXON_IDS matching SPECIES', () => {
    for (const species of SPECIES) {
      expect(TAXON_IDS[species.id]).toBe(species.taxonId);
    }
  });

  it('every species belongs to exactly one valid type group', () => {
    const validIds = new Set(TYPE_GROUPS.map((g) => g.id));
    for (const species of SPECIES) {
      expect(validIds.has(species.typeGroup)).toBe(true);
    }
  });

  it('every species belongs to exactly one valid habitat', () => {
    const validIds = new Set(HABITAT_GROUPS.map((g) => g.id));
    for (const species of SPECIES) {
      expect(validIds.has(species.habitat)).toBe(true);
    }
  });

  it('every species has a valid migration status', () => {
    const validIds = new Set(MIGRATION_GROUPS.map((g) => g.id));
    for (const species of SPECIES) {
      expect(validIds.has(species.migration)).toBe(true);
    }
  });

  it('type groups partition the roster with no leftovers', () => {
    const total = TYPE_GROUPS.reduce(
      (sum, g) => sum + SPECIES.filter((s) => s.typeGroup === g.id).length,
      0
    );
    expect(total).toBe(SPECIES_COUNT);
  });

  it('habitat groups partition the roster with no leftovers', () => {
    const total = HABITAT_GROUPS.reduce(
      (sum, g) => sum + SPECIES.filter((s) => s.habitat === g.id).length,
      0
    );
    expect(total).toBe(SPECIES_COUNT);
  });
});

describe('formatSpeciesName', () => {
  it('replaces underscores with spaces', () => {
    expect(formatSpeciesName('Barn_Owl')).toBe('Barn Owl');
    expect(formatSpeciesName('Black_headed_Gull')).toBe('Black headed Gull');
  });

  it('strips a leading numeric prefix', () => {
    expect(formatSpeciesName('3.Barn_Owl')).toBe('Barn Owl');
  });

  it('returns an empty string for falsy input', () => {
    expect(formatSpeciesName(null)).toBe('');
    expect(formatSpeciesName(undefined)).toBe('');
    expect(formatSpeciesName('')).toBe('');
  });
});
