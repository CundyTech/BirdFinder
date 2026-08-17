import { configureStore } from '@reduxjs/toolkit';
import filmReducer, {
  hydrateFilm,
  spendFilm,
  claimDailyRefill,
  claimTrophyRewards,
  grantAdReward,
  DAILY_REFILL_AMOUNT,
  TROPHY_REWARD_AMOUNT,
  AD_REWARD_AMOUNT,
  AD_DAILY_CAP,
} from './filmSlice';
import * as filmStorage from '../services/filmStorage';

// Storage itself is covered by filmStorage.test.js — mocking it here
// isolates the reducer/thunk wiring.
jest.mock('../services/filmStorage');

function makeStore() {
  return configureStore({ reducer: { film: filmReducer } });
}

describe('filmSlice', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('starts unhydrated with a zero balance (the real value only arrives via hydrateFilm)', () => {
    const store = makeStore();
    expect(store.getState().film).toEqual({
      balance: 0,
      lastRefillDate: null,
      claimedTrophyKeys: [],
      adsWatchedToday: 0,
      adsWatchedDate: null,
      hydrated: false,
    });
  });

  it('hydrateFilm loads state from storage and marks the slice hydrated', async () => {
    const saved = {
      balance: 5,
      lastRefillDate: '2026-01-01',
      claimedTrophyKeys: ['rarity:Rare'],
      adsWatchedToday: 2,
      adsWatchedDate: '2026-01-01',
    };
    filmStorage.loadFilmState.mockResolvedValue(saved);

    const store = makeStore();
    await store.dispatch(hydrateFilm());

    expect(store.getState().film).toEqual({ ...saved, hydrated: true });
  });

  it('spendFilm replaces state with the storage result', async () => {
    filmStorage.spendFilm.mockResolvedValue({ balance: 4, lastRefillDate: null, claimedTrophyKeys: [] });

    const store = makeStore();
    await store.dispatch(spendFilm());

    expect(store.getState().film.balance).toBe(4);
  });

  it('spendFilm leaves state untouched when storage reports nothing to spend', async () => {
    filmStorage.spendFilm.mockResolvedValue(null);

    const store = makeStore();
    await store.dispatch(hydrateFilm.fulfilled({ balance: 0, lastRefillDate: null, claimedTrophyKeys: [] }));
    await store.dispatch(spendFilm());

    expect(store.getState().film.balance).toBe(0);
  });

  it('claimDailyRefill calls storage with the configured amount and today\'s date', async () => {
    filmStorage.claimDailyRefill.mockResolvedValue({
      balance: 7,
      lastRefillDate: '2026-08-17',
      claimedTrophyKeys: [],
    });

    const store = makeStore();
    await store.dispatch(claimDailyRefill());

    expect(filmStorage.claimDailyRefill).toHaveBeenCalledWith(
      DAILY_REFILL_AMOUNT,
      expect.stringMatching(/^\d{4}-\d{2}-\d{2}$/)
    );
    expect(store.getState().film.balance).toBe(7);
  });

  it('claimTrophyRewards calls storage with the configured amount and forwards the unlocked keys', async () => {
    filmStorage.claimTrophyRewards.mockResolvedValue({
      state: { balance: 15, lastRefillDate: null, claimedTrophyKeys: ['rarity:Very rare'] },
      newlyClaimed: ['rarity:Very rare'],
    });

    const store = makeStore();
    await store.dispatch(claimTrophyRewards(['rarity:Very rare']));

    expect(filmStorage.claimTrophyRewards).toHaveBeenCalledWith(TROPHY_REWARD_AMOUNT, ['rarity:Very rare']);
    expect(store.getState().film).toMatchObject({ balance: 15, claimedTrophyKeys: ['rarity:Very rare'] });
  });

  it('grantAdReward calls storage with the configured amount/cap and applies the result', async () => {
    filmStorage.grantAdReward.mockResolvedValue({
      state: { balance: 10, lastRefillDate: null, claimedTrophyKeys: [], adsWatchedToday: 1, adsWatchedDate: '2026-08-17' },
      granted: true,
    });

    const store = makeStore();
    await store.dispatch(grantAdReward());

    expect(filmStorage.grantAdReward).toHaveBeenCalledWith(
      AD_REWARD_AMOUNT,
      AD_DAILY_CAP,
      expect.stringMatching(/^\d{4}-\d{2}-\d{2}$/)
    );
    expect(store.getState().film.balance).toBe(10);
  });
});
