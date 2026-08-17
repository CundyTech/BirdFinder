import AsyncStorage from '@react-native-async-storage/async-storage';
import { getCachedSpeciesInfo, setCachedSpeciesInfo, __resetCacheForTests } from './birdInfoCache';

const STORAGE_KEY = 'birdInfo.cache.v2';

describe('birdInfoCache', () => {
  beforeEach(async () => {
    await AsyncStorage.clear();
    __resetCacheForTests();
  });

  it('returns undefined for a species that has never been cached', async () => {
    expect(await getCachedSpeciesInfo('Common_Kestrel')).toBeUndefined();
  });

  it('stores and retrieves a species entry', async () => {
    const data = { commonName: 'Common Kestrel', family: 'Falconidae' };

    await setCachedSpeciesInfo('Common_Kestrel', data);

    expect(await getCachedSpeciesInfo('Common_Kestrel')).toEqual(data);
  });

  it('distinguishes a cached null from never-cached (undefined) — callers rely on this to avoid re-fetching a genuine "no match"', async () => {
    await setCachedSpeciesInfo('Unknown_Species', null);

    expect(await getCachedSpeciesInfo('Unknown_Species')).toBeNull();
  });

  it('persists to AsyncStorage under the versioned key', async () => {
    await setCachedSpeciesInfo('Barn_Owl', { commonName: 'Barn Owl' });

    const raw = await AsyncStorage.getItem(STORAGE_KEY);
    expect(JSON.parse(raw)).toEqual({ Barn_Owl: { commonName: 'Barn Owl' } });
  });

  it('reads whatever was already in AsyncStorage before this module touched it', async () => {
    await AsyncStorage.setItem(STORAGE_KEY, JSON.stringify({ Rook: { commonName: 'Rook' } }));
    __resetCacheForTests(); // the seeded write above happened outside this module's memoized read

    expect(await getCachedSpeciesInfo('Rook')).toEqual({ commonName: 'Rook' });
  });

  it('recovers from corrupted JSON instead of throwing', async () => {
    await AsyncStorage.setItem(STORAGE_KEY, 'not valid json{');
    __resetCacheForTests();

    expect(await getCachedSpeciesInfo('Rook')).toBeUndefined();
  });

  it('does not clobber other cached species when writing one', async () => {
    await setCachedSpeciesInfo('Barn_Owl', { commonName: 'Barn Owl' });
    await setCachedSpeciesInfo('Mallard', { commonName: 'Mallard' });

    expect(await getCachedSpeciesInfo('Barn_Owl')).toEqual({ commonName: 'Barn Owl' });
    expect(await getCachedSpeciesInfo('Mallard')).toEqual({ commonName: 'Mallard' });
  });
});
