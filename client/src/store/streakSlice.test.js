import { configureStore } from '@reduxjs/toolkit';
import streakReducer, { hydrateStreak, recordActivity } from './streakSlice';
import * as streakStorage from '../services/streakStorage';
import * as rewardsStorage from '../services/rewardsStorage';

// Both storage modules are covered by their own *.test.js files — mocking
// them here isolates the reducer/thunk wiring, including the cross-slice
// read of rewards.streakProtectionTokens that recordActivity depends on.
jest.mock('../services/streakStorage');
jest.mock('../services/rewardsStorage');

function makeStore(streakProtectionTokens = 0) {
  return configureStore({
    reducer: {
      streak: streakReducer,
      rewards: () => ({ streakProtectionTokens }),
    },
  });
}

describe('streakSlice', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('starts unhydrated at zero', () => {
    const store = makeStore();
    expect(store.getState().streak).toEqual({
      currentStreak: 0,
      longestStreak: 0,
      lastActiveDate: null,
      hydrated: false,
    });
  });

  it('hydrateStreak loads state from storage and marks the slice hydrated', async () => {
    const saved = { currentStreak: 3, longestStreak: 5, lastActiveDate: '2026-01-01' };
    streakStorage.loadStreakState.mockResolvedValue(saved);

    const store = makeStore();
    await store.dispatch(hydrateStreak());

    expect(store.getState().streak).toEqual({ ...saved, hydrated: true });
  });

  it('recordActivity applies the storage result and does not touch protection when none was used', async () => {
    streakStorage.recordActivity.mockResolvedValue({
      state: { currentStreak: 1, longestStreak: 1, lastActiveDate: '2026-01-01' },
      protectionUsed: false,
    });

    const store = makeStore(2);
    await store.dispatch(recordActivity());

    expect(store.getState().streak.currentStreak).toBe(1);
    expect(rewardsStorage.consumeStreakProtection).not.toHaveBeenCalled();
  });

  it('recordActivity checks rewards.streakProtectionTokens and consumes a token when the streak used protection', async () => {
    streakStorage.recordActivity.mockResolvedValue({
      state: { currentStreak: 4, longestStreak: 4, lastActiveDate: '2026-01-05' },
      protectionUsed: true,
    });

    const store = makeStore(1);
    await store.dispatch(recordActivity());

    expect(streakStorage.recordActivity).toHaveBeenCalledWith(expect.stringMatching(/^\d{4}-\d{2}-\d{2}$/), true);
    expect(rewardsStorage.consumeStreakProtection).toHaveBeenCalled();
  });

  it('passes hasProtection as false when no protection tokens are available', async () => {
    streakStorage.recordActivity.mockResolvedValue({
      state: { currentStreak: 1, longestStreak: 1, lastActiveDate: '2026-01-01' },
      protectionUsed: false,
    });

    const store = makeStore(0);
    await store.dispatch(recordActivity());

    expect(streakStorage.recordActivity).toHaveBeenCalledWith(expect.any(String), false);
  });
});
