import { createAsyncThunk, createSlice } from '@reduxjs/toolkit';
import {
  loadFilmState,
  spendFilm as spendFilmInStorage,
  claimDailyRefill as claimDailyRefillInStorage,
  claimTrophyRewards as claimTrophyRewardsInStorage,
} from '../services/filmStorage';

// Economy tuning knobs. Ad rewards join this list in phase 2, once a real
// rewarded-ad SDK exists to call grantAdReward from.
export const DAILY_REFILL_AMOUNT = 2;
export const TROPHY_REWARD_AMOUNT = 10;

function todayKey() {
  return new Date().toISOString().slice(0, 10);
}

export const hydrateFilm = createAsyncThunk('film/hydrate', async () => {
  return loadFilmState();
});

export const spendFilm = createAsyncThunk('film/spend', async () => {
  return spendFilmInStorage();
});

export const claimDailyRefill = createAsyncThunk('film/claimDailyRefill', async () => {
  return claimDailyRefillInStorage(DAILY_REFILL_AMOUNT, todayKey());
});

// arg: array of "categoryId:trophyLabel" keys for every trophy currently
// unlocked (not just newly-unlocked ones — the storage layer diffs against
// what's already been claimed).
export const claimTrophyRewards = createAsyncThunk(
  'film/claimTrophyRewards',
  async (unlockedTrophyKeys) => {
    return claimTrophyRewardsInStorage(TROPHY_REWARD_AMOUNT, unlockedTrophyKeys);
  }
);

const filmSlice = createSlice({
  name: 'film',
  initialState: {
    balance: 0,
    lastRefillDate: null,
    claimedTrophyKeys: [],
    hydrated: false,
  },
  reducers: {},
  extraReducers: (builder) => {
    builder
      .addCase(hydrateFilm.fulfilled, (state, action) => {
        Object.assign(state, action.payload);
        state.hydrated = true;
      })
      .addCase(spendFilm.fulfilled, (state, action) => {
        // null means the balance was already zero — nothing to apply.
        if (action.payload) Object.assign(state, action.payload);
      })
      .addCase(claimDailyRefill.fulfilled, (state, action) => {
        Object.assign(state, action.payload);
      })
      .addCase(claimTrophyRewards.fulfilled, (state, action) => {
        Object.assign(state, action.payload.state);
      });
  },
});

export default filmSlice.reducer;
