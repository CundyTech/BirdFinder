import AsyncStorage from '@react-native-async-storage/async-storage';

const STORAGE_KEY = 'birdInfo.cache.v1';

// Loaded once and shared: on a cold start, ~60 species tiles all call
// getCachedSpeciesInfo at once — this ensures they await one AsyncStorage
// read instead of 60, and all mutate the same in-memory object so
// concurrent writes converge instead of clobbering each other.
let cachePromise = null;

function loadCache() {
  if (!cachePromise) {
    cachePromise = AsyncStorage.getItem(STORAGE_KEY).then((raw) => {
      if (!raw) return {};
      try {
        const parsed = JSON.parse(raw);
        return parsed && typeof parsed === 'object' ? parsed : {};
      } catch {
        return {};
      }
    });
  }
  return cachePromise;
}

// undefined = never fetched (should look it up); anything else, including
// null, is a cached result that shouldn't trigger another network call.
export async function getCachedSpeciesInfo(commonName) {
  const cache = await loadCache();
  return Object.prototype.hasOwnProperty.call(cache, commonName) ? cache[commonName] : undefined;
}

export async function setCachedSpeciesInfo(commonName, data) {
  const cache = await loadCache();
  cache[commonName] = data;
  await AsyncStorage.setItem(STORAGE_KEY, JSON.stringify(cache));
}
