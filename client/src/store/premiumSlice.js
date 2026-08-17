import { createAsyncThunk, createSlice } from '@reduxjs/toolkit';
import { loadPremiumState, setUnlockedForever as setUnlockedForeverInStorage } from '../services/premiumStorage';

export const hydratePremium = createAsyncThunk('premium/hydrate', async () => {
  return loadPremiumState();
});

// Dispatched from the IAP purchase-updated listener once a purchase is
// verified as completed — see services/purchases.js. Never called directly
// from a "buy button" onPress; that only starts the purchase flow.
export const unlockForever = createAsyncThunk('premium/unlockForever', async () => {
  return setUnlockedForeverInStorage();
});

const premiumSlice = createSlice({
  name: 'premium',
  initialState: {
    unlockedForever: false,
    hydrated: false,
  },
  reducers: {},
  extraReducers: (builder) => {
    builder
      .addCase(hydratePremium.fulfilled, (state, action) => {
        state.unlockedForever = action.payload.unlockedForever;
        state.hydrated = true;
      })
      .addCase(unlockForever.fulfilled, (state, action) => {
        state.unlockedForever = action.payload.unlockedForever;
      });
  },
});

export default premiumSlice.reducer;
