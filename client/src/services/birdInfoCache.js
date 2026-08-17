import AsyncStorage from '@react-native-async-storage/async-storage';

// v2: keyed by our internal speciesId (e.g. "Common_Kestrel") instead of
// a display name — bumped so any old entries (including ones from the
// previous name-search flow that could get stuck cached with missing
// family/order/summary after a partial failure) are cleanly abandoned
// rather than mixed in under different-looking keys.
const STORAGE_KEY = 'birdInfo.cache.v2';

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
export async function getCachedSpeciesInfo(speciesId) {
  const cache = await loadCache();
  return Object.prototype.hasOwnProperty.call(cache, speciesId) ? cache[speciesId] : undefined;
}

export async function setCachedSpeciesInfo(speciesId, data) {
  const cache = await loadCache();
  cache[speciesId] = data;
  await AsyncStorage.setItem(STORAGE_KEY, JSON.stringify(cache));
}
