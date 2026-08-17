import AsyncStorage from '@react-native-async-storage/async-storage';

const STORAGE_KEY = 'film.state.v1';

// Starting balance lives here (not filmSlice.js) since it's part of "what
// does a fresh save look like", not a game-balance tuning knob — those
// (refill/reward amounts) live in filmSlice.js next to the thunks that use
// them.
export const STARTING_BALANCE = 5;

function defaultState() {
  return {
    balance: STARTING_BALANCE,
    lastRefillDate: null,
    claimedTrophyKeys: [],
    adsWatchedToday: 0,
    adsWatchedDate: null,
  };
}

function sanitize(parsed) {
  if (!parsed || typeof parsed !== 'object') return defaultState();
  return {
    balance: typeof parsed.balance === 'number' ? parsed.balance : STARTING_BALANCE,
    lastRefillDate: typeof parsed.lastRefillDate === 'string' ? parsed.lastRefillDate : null,
    claimedTrophyKeys: Array.isArray(parsed.claimedTrophyKeys) ? parsed.claimedTrophyKeys : [],
    adsWatchedToday: typeof parsed.adsWatchedToday === 'number' ? parsed.adsWatchedToday : 0,
    adsWatchedDate: typeof parsed.adsWatchedDate === 'string' ? parsed.adsWatchedDate : null,
  };
}

export async function loadFilmState() {
  const raw = await AsyncStorage.getItem(STORAGE_KEY);
  if (!raw) return defaultState();
  try {
    return sanitize(JSON.parse(raw));
  } catch {
    return defaultState();
  }
}

async function saveFilmState(state) {
  await AsyncStorage.setItem(STORAGE_KEY, JSON.stringify(state));
}

// Spends one Film. Returns the updated state, or null if the balance was
// already zero — callers should treat null as "nothing to spend" rather
// than letting the balance go negative. Gating (blocking the action that
// would have spent it) is the UI's job; this is just the ledger.
export async function spendFilm() {
  const state = await loadFilmState();
  if (state.balance <= 0) return null;
  const updated = { ...state, balance: state.balance - 1 };
  await saveFilmState(updated);
  return updated;
}

// Grants the daily refill if `today` hasn't already been claimed. Returns
// the state unchanged if it has — safe to call on every app launch.
export async function claimDailyRefill(amount, today) {
  const state = await loadFilmState();
  if (state.lastRefillDate === today) return state;
  const updated = { ...state, balance: state.balance + amount, lastRefillDate: today };
  await saveFilmState(updated);
  return updated;
}

// Grants `amountPerTrophy` for each key in unlockedTrophyKeys not already
// in claimedTrophyKeys. Callers can (and do) pass the full current list of
// unlocked trophies every time — already-claimed ones are filtered out
// here, so this is idempotent and safe to call repeatedly as trophy state
// is recomputed.
export async function claimTrophyRewards(amountPerTrophy, unlockedTrophyKeys) {
  const state = await loadFilmState();
  const newKeys = unlockedTrophyKeys.filter((key) => !state.claimedTrophyKeys.includes(key));
  if (newKeys.length === 0) return { state, newlyClaimed: [] };

  const updated = {
    ...state,
    balance: state.balance + amountPerTrophy * newKeys.length,
    claimedTrophyKeys: [...state.claimedTrophyKeys, ...newKeys],
  };
  await saveFilmState(updated);
  return { state: updated, newlyClaimed: newKeys };
}

// Grants a rewarded-ad bonus, up to dailyCap watches per calendar day.
// Returns { state, granted } — granted is false once the cap's hit for
// `today`, so the UI can disable the watch-ad option without a separate
// "how many have I watched" query.
export async function grantAdReward(amount, dailyCap, today) {
  const state = await loadFilmState();
  const watchedToday = state.adsWatchedDate === today ? state.adsWatchedToday : 0;

  if (watchedToday >= dailyCap) {
    return { state, granted: false };
  }

  const updated = {
    ...state,
    balance: state.balance + amount,
    adsWatchedToday: watchedToday + 1,
    adsWatchedDate: today,
  };
  await saveFilmState(updated);
  return { state: updated, granted: true };
}
