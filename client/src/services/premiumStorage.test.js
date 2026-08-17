import AsyncStorage from '@react-native-async-storage/async-storage';
import { loadPremiumState, setUnlockedForever } from './premiumStorage';

const STORAGE_KEY = 'premium.state.v1';

describe('premiumStorage', () => {
  beforeEach(async () => {
    await AsyncStorage.clear();
  });

  it('defaults to not unlocked', async () => {
    expect(await loadPremiumState()).toEqual({ unlockedForever: false });
  });

  it('recovers from corrupted JSON instead of throwing', async () => {
    await AsyncStorage.setItem(STORAGE_KEY, 'not json{');
    expect(await loadPremiumState()).toEqual({ unlockedForever: false });
  });

  it('setUnlockedForever persists the unlock', async () => {
    const result = await setUnlockedForever();

    expect(result).toEqual({ unlockedForever: true });
    expect(await loadPremiumState()).toEqual({ unlockedForever: true });
  });
});
