import { BIRD_PROFILES, MIGRATION_ROUTES } from './birdProfiles';
import { SPECIES } from './species';

// birdProfiles.js is hand-curated content keyed by speciesId, maintained
// separately from species.js — nothing enforces they stay in sync except
// tests like these.
describe('BIRD_PROFILES', () => {
  it('has a complete profile for every species in the roster', () => {
    for (const species of SPECIES) {
      const profile = BIRD_PROFILES[species.id];
      expect(profile).toBeDefined();
      expect(typeof profile.fact).toBe('string');
      expect(profile.fact.length).toBeGreaterThan(0);
      expect(typeof profile.prey).toBe('string');
      expect(profile.prey.length).toBeGreaterThan(0);
      expect(typeof profile.predators).toBe('string');
      expect(profile.predators.length).toBeGreaterThan(0);
    }
  });

  it('has no orphaned entries for species that no longer exist', () => {
    const speciesIds = new Set(SPECIES.map((s) => s.id));
    for (const id of Object.keys(BIRD_PROFILES)) {
      expect(speciesIds.has(id)).toBe(true);
    }
  });
});

describe('MIGRATION_ROUTES', () => {
  it('has a route for exactly the species marked as summer visitors, no more, no less', () => {
    const summerVisitorIds = SPECIES.filter((s) => s.migration === 'summer')
      .map((s) => s.id)
      .sort();
    const routeIds = Object.keys(MIGRATION_ROUTES).sort();
    expect(routeIds).toEqual(summerVisitorIds);
  });

  it('has no route for a resident species', () => {
    const residentIds = new Set(SPECIES.filter((s) => s.migration === 'resident').map((s) => s.id));
    for (const id of Object.keys(MIGRATION_ROUTES)) {
      expect(residentIds.has(id)).toBe(false);
    }
  });

  it('every route describes both a breeding and a wintering location', () => {
    for (const route of Object.values(MIGRATION_ROUTES)) {
      expect(typeof route.breeding).toBe('string');
      expect(route.breeding.length).toBeGreaterThan(0);
      expect(typeof route.wintering).toBe('string');
      expect(route.wintering.length).toBeGreaterThan(0);
    }
  });
});
