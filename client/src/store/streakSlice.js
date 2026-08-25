import { createAsyncThunk, createSlice } from '@reduxjs/toolkit';
import { loadStreakState, recordActivity as recordActivityInStorage } from '../services/streakStorage';
import { consumeStreakProtection } from '../services/rewardsStorage';

function todayKey() {
  return new Date().toISOString().slice(0, 10);
}

export const hydrateStreak = createAsyncThunk('streak/hydrate', async () => {
  return loadStreakState();
});

// Dispatched once per recorded sighting (see HomeScreen.js). A missed day
// covered by streak protection also spends the token here, in the same
// thunk — rewardsSlice.js mirrors that spend into its own state by
// listening for this thunk's fulfilled action (see rewardsSlice.js).
export const recordActivity = createAsyncThunk('streak/recordActivity', async (_, { getState }) => {
  const hasProtection = getState().rewards.streakProtectionTokens > 0;
  const result = await recordActivityInStorage(todayKey(), hasProtection);
  if (result.protectionUsed) {
    await consumeStreakProtection();
  }
  return result;
});

const streakSlice = createSlice({
  name: 'streak',
  initialState: {
    currentStreak: 0,
    longestStreak: 0,
    lastActiveDate: null,
    hydrated: false,
  },
  reducers: {},
  extraReducers: (builder) => {
    builder
      .addCase(hydrateStreak.fulfilled, (state, action) => {
        Object.assign(state, action.payload);
        state.hydrated = true;
      })
      .addCase(recordActivity.fulfilled, (state, action) => {
        Object.assign(state, action.payload.state);
      });
  },
});

export default streakSlice.reducer;
