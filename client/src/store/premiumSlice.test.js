import { configureStore } from '@reduxjs/toolkit';
import premiumReducer, { hydratePremium, unlockForever } from './premiumSlice';
import * as premiumStorage from '../services/premiumStorage';

jest.mock('../services/premiumStorage');

function makeStore() {
  return configureStore({ reducer: { premium: premiumReducer } });
}

describe('premiumSlice', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('starts locked and unhydrated', () => {
    const store = makeStore();
    expect(store.getState().premium).toEqual({ unlockedForever: false, hydrated: false });
  });

  it('hydratePremium loads state from storage and marks hydrated', async () => {
    premiumStorage.loadPremiumState.mockResolvedValue({ unlockedForever: true });

    const store = makeStore();
    await store.dispatch(hydratePremium());

    expect(store.getState().premium).toEqual({ unlockedForever: true, hydrated: true });
  });

  it('unlockForever flips the flag once storage confirms it', async () => {
    premiumStorage.setUnlockedForever.mockResolvedValue({ unlockedForever: true });

    const store = makeStore();
    await store.dispatch(unlockForever());

    expect(store.getState().premium.unlockedForever).toBe(true);
  });
});
