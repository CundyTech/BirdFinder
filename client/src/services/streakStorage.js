import AsyncStorage from '@react-native-async-storage/async-storage';
import { createSemaphore } from './asyncSemaphore';

const STORAGE_KEY = 'streak.state.v1';

// See rewardsStorage.js for why this exists — same read-modify-write
// storage pattern, serialized against overlapping calls.
const withLock = createSemaphore(1);

function defaultState() {
  return {
    currentStreak: 0,
    longestStreak: 0,
    lastActiveDate: null,
  };
}

function sanitize(parsed) {
  if (!parsed || typeof parsed !== 'object') return defaultState();
  return {
    currentStreak: typeof parsed.currentStreak === 'number' ? parsed.currentStreak : 0,
    longestStreak: typeof parsed.longestStreak === 'number' ? parsed.longestStreak : 0,
    lastActiveDate: typeof parsed.lastActiveDate === 'string' ? parsed.lastActiveDate : null,
  };
}

export async function loadStreakState() {
  const raw = await AsyncStorage.getItem(STORAGE_KEY);
  if (!raw) return defaultState();
  try {
    return sanitize(JSON.parse(raw));
  } catch {
    return defaultState();
  }
}

async function saveStreakState(state) {
  await AsyncStorage.setItem(STORAGE_KEY, JSON.stringify(state));
}

// 'YYYY-MM-DD' string difference in whole days — cheaper and less
// timezone-sensitive than parsing into Date objects for arithmetic.
function daysBetween(earlier, later) {
  const a = Date.UTC(...earlier.split('-').map(Number));
  const b = Date.UTC(...later.split('-').map(Number));
  return Math.round((b - a) / 86400000);
}

// Called once per day the user records a sighting. A gap of exactly one
// missed day is covered by streak protection if available (consumed by the
// caller — see streakSlice.js); any bigger gap resets the streak regardless.
export function recordActivity(today, hasProtection) {
  return withLock(async () => {
    const state = await loadStreakState();

    if (state.lastActiveDate === today) {
      return { state, protectionUsed: false };
    }

    let nextStreak;
    let protectionUsed = false;

    if (state.lastActiveDate === null) {
      nextStreak = 1;
    } else {
      const gap = daysBetween(state.lastActiveDate, today);
      if (gap === 1) {
        nextStreak = state.currentStreak + 1;
      } else if (gap === 2 && hasProtection) {
        nextStreak = state.currentStreak + 1;
        protectionUsed = true;
      } else {
        nextStreak = 1;
      }
    }

    const updated = {
      currentStreak: nextStreak,
      longestStreak: Math.max(state.longestStreak, nextStreak),
      lastActiveDate: today,
    };
    await saveStreakState(updated);
    return { state: updated, protectionUsed };
  });
}
