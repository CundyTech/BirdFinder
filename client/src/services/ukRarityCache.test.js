import AsyncStorage from '@react-native-async-storage/async-storage';
import { getCachedRarityMap, setCachedRarityMap } from './ukRarityCache';

const STORAGE_KEY = 'ukRarity.cache.v1';

describe('ukRarityCache', () => {
  beforeEach(async () => {
    await AsyncStorage.clear();
  });

  it('returns undefined when nothing has been cached yet', async () => {
    expect(await getCachedRarityMap()).toBeUndefined();
  });

  it('stores and retrieves the whole rarity map', async () => {
    const map = { Barn_Owl: 2033, Mallard: 43687 };

    await setCachedRarityMap(map);

    expect(await getCachedRarityMap()).toEqual(map);
  });

  it('persists under the expected storage key', async () => {
    await setCachedRarityMap({ Barn_Owl: 2033 });

    const raw = await AsyncStorage.getItem(STORAGE_KEY);
    expect(JSON.parse(raw)).toEqual({ Barn_Owl: 2033 });
  });

  it('recovers from corrupted JSON instead of throwing', async () => {
    await AsyncStorage.setItem(STORAGE_KEY, 'not valid json{');
    expect(await getCachedRarityMap()).toBeUndefined();
  });

  it('rejects a non-object JSON value (e.g. a bare string or number)', async () => {
    await AsyncStorage.setItem(STORAGE_KEY, JSON.stringify('just a string'));
    expect(await getCachedRarityMap()).toBeUndefined();
  });
});
