import AsyncStorage from '@react-native-async-storage/async-storage';

const STORAGE_KEY = 'premium.state.v1';

export async function loadPremiumState() {
  const raw = await AsyncStorage.getItem(STORAGE_KEY);
  if (!raw) return { unlockedForever: false };
  try {
    const parsed = JSON.parse(raw);
    return { unlockedForever: Boolean(parsed?.unlockedForever) };
  } catch {
    return { unlockedForever: false };
  }
}

// One-way flip — there's no code path that un-sets this. It's only ever
// set from a verified purchase-completed event (see services/purchases.js),
// never speculatively.
export async function setUnlockedForever() {
  const state = { unlockedForever: true };
  await AsyncStorage.setItem(STORAGE_KEY, JSON.stringify(state));
  return state;
}
