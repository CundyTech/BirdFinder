import AsyncStorage from '@react-native-async-storage/async-storage';

const STORAGE_KEY = 'ukRarity.cache.v1';

export async function getCachedRarityMap() {
  const raw = await AsyncStorage.getItem(STORAGE_KEY);
  if (!raw) return undefined;
  try {
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === 'object' ? parsed : undefined;
  } catch {
    return undefined;
  }
}

export async function setCachedRarityMap(map) {
  await AsyncStorage.setItem(STORAGE_KEY, JSON.stringify(map));
}
