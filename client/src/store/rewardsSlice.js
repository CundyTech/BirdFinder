import { createAsyncThunk, createSlice } from '@reduxjs/toolkit';
import {
  loadRewardsState,
  spendReroll as spendRerollInStorage,
  equipFrame as equipFrameInStorage,
  grantMilestoneRewards as grantMilestoneRewardsInStorage,
} from '../services/rewardsStorage';
import { recordActivity } from './streakSlice';

export const hydrateRewards = createAsyncThunk('rewards/hydrate', async () => {
  return loadRewardsState();
});

export const spendReroll = createAsyncThunk('rewards/spendReroll', async () => {
  return spendRerollInStorage();
});

export const equipFrame = createAsyncThunk('rewards/equipFrame', async (frameId) => {
  return equipFrameInStorage(frameId);
});

// arg: array of { key, reward } for every milestone trophy currently
// unlocked (not just newly-unlocked ones — the storage layer diffs against
// what's already been claimed, same convention as filmSlice's
// claimTrophyRewards).
export const grantMilestoneRewards = createAsyncThunk(
  'rewards/grantMilestoneRewards',
  async (unlockedMilestones) => {
    return grantMilestoneRewardsInStorage(unlockedMilestones);
  }
);

const rewardsSlice = createSlice({
  name: 'rewards',
  initialState: {
    rerollTokens: 0,
    streakProtectionTokens: 0,
    ownedFrameIds: [],
    equippedFrameId: null,
    deepDiveLoreUnlocked: false,
    claimedMilestoneKeys: [],
    hydrated: false,
  },
  reducers: {},
  extraReducers: (builder) => {
    builder
      .addCase(hydrateRewards.fulfilled, (state, action) => {
        Object.assign(state, action.payload);
        state.hydrated = true;
      })
      .addCase(spendReroll.fulfilled, (state, action) => {
        // null means there was nothing to spend — nothing to apply.
        if (action.payload) Object.assign(state, action.payload);
      })
      .addCase(equipFrame.fulfilled, (state, action) => {
        Object.assign(state, action.payload);
      })
      .addCase(grantMilestoneRewards.fulfilled, (state, action) => {
        Object.assign(state, action.payload.state);
      })
      // streakSlice.recordActivity spends a streak protection token
      // directly in storage when it covers a missed day (see
      // streakSlice.js) — mirror that spend into this slice's in-memory
      // state so the UI reflects it without a redundant re-hydrate.
      .addCase(recordActivity.fulfilled, (state, action) => {
        if (action.payload.protectionUsed) {
          state.streakProtectionTokens = Math.max(0, state.streakProtectionTokens - 1);
        }
      });
  },
});

export default rewardsSlice.reducer;
