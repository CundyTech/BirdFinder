import { configureStore } from '@reduxjs/toolkit';
import rewardsReducer, {
  hydrateRewards,
  spendReroll,
  equipFrame,
  grantMilestoneRewards,
} from './rewardsSlice';
import { recordActivity } from './streakSlice';
import * as rewardsStorage from '../services/rewardsStorage';

// Storage itself is covered by rewardsStorage.test.js — mocking it here
// isolates the reducer/thunk wiring.
jest.mock('../services/rewardsStorage');

function makeStore() {
  return configureStore({ reducer: { rewards: rewardsReducer } });
}

describe('rewardsSlice', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('starts unhydrated with empty balances', () => {
    const store = makeStore();
    expect(store.getState().rewards).toEqual({
      rerollTokens: 0,
      streakProtectionTokens: 0,
      ownedFrameIds: [],
      equippedFrameId: null,
      deepDiveLoreUnlocked: false,
      claimedMilestoneKeys: [],
      hydrated: false,
    });
  });

  it('hydrateRewards loads state from storage and marks the slice hydrated', async () => {
    const saved = {
      rerollTokens: 2,
      streakProtectionTokens: 1,
      ownedFrameIds: ['bronze-feather'],
      equippedFrameId: 'bronze-feather',
      deepDiveLoreUnlocked: true,
      claimedMilestoneKeys: ['milestones:First Reroll'],
    };
    rewardsStorage.loadRewardsState.mockResolvedValue(saved);

    const store = makeStore();
    await store.dispatch(hydrateRewards());

    expect(store.getState().rewards).toEqual({ ...saved, hydrated: true });
  });

  it('spendReroll applies the storage result', async () => {
    rewardsStorage.spendReroll.mockResolvedValue({
      rerollTokens: 0,
      streakProtectionTokens: 0,
      ownedFrameIds: [],
      equippedFrameId: null,
      deepDiveLoreUnlocked: false,
      claimedMilestoneKeys: [],
    });

    const store = makeStore();
    await store.dispatch(spendReroll());

    expect(store.getState().rewards.rerollTokens).toBe(0);
  });

  it('spendReroll leaves state untouched when storage reports nothing to spend', async () => {
    rewardsStorage.spendReroll.mockResolvedValue(null);

    const store = makeStore();
    await store.dispatch(
      hydrateRewards.fulfilled({
        rerollTokens: 3,
        streakProtectionTokens: 0,
        ownedFrameIds: [],
        equippedFrameId: null,
        deepDiveLoreUnlocked: false,
        claimedMilestoneKeys: [],
      })
    );
    await store.dispatch(spendReroll());

    expect(store.getState().rewards.rerollTokens).toBe(3);
  });

  it('equipFrame applies the storage result', async () => {
    rewardsStorage.equipFrame.mockResolvedValue({
      rerollTokens: 0,
      streakProtectionTokens: 0,
      ownedFrameIds: ['golden-wing'],
      equippedFrameId: 'golden-wing',
      deepDiveLoreUnlocked: false,
      claimedMilestoneKeys: [],
    });

    const store = makeStore();
    await store.dispatch(equipFrame('golden-wing'));

    expect(rewardsStorage.equipFrame).toHaveBeenCalledWith('golden-wing');
    expect(store.getState().rewards.equippedFrameId).toBe('golden-wing');
  });

  it('grantMilestoneRewards forwards the milestone list and applies the storage result', async () => {
    const milestones = [{ key: 'milestones:First Reroll', reward: { type: 'reroll', amount: 1 } }];
    rewardsStorage.grantMilestoneRewards.mockResolvedValue({
      state: {
        rerollTokens: 1,
        streakProtectionTokens: 0,
        ownedFrameIds: [],
        equippedFrameId: null,
        deepDiveLoreUnlocked: false,
        claimedMilestoneKeys: ['milestones:First Reroll'],
      },
      newlyClaimed: ['milestones:First Reroll'],
    });

    const store = makeStore();
    await store.dispatch(grantMilestoneRewards(milestones));

    expect(rewardsStorage.grantMilestoneRewards).toHaveBeenCalledWith(milestones);
    expect(store.getState().rewards.rerollTokens).toBe(1);
  });

  it('mirrors a streak-protection spend when streakSlice.recordActivity reports it was used', async () => {
    const store = makeStore();
    await store.dispatch(
      hydrateRewards.fulfilled({
        rerollTokens: 0,
        streakProtectionTokens: 2,
        ownedFrameIds: [],
        equippedFrameId: null,
        deepDiveLoreUnlocked: false,
        claimedMilestoneKeys: [],
      })
    );

    store.dispatch(
      recordActivity.fulfilled(
        { state: { currentStreak: 4, longestStreak: 4, lastActiveDate: '2026-01-05' }, protectionUsed: true },
        'test-request-id'
      )
    );

    expect(store.getState().rewards.streakProtectionTokens).toBe(1);
  });

  it('does not change streakProtectionTokens when recordActivity did not use protection', async () => {
    const store = makeStore();
    await store.dispatch(
      hydrateRewards.fulfilled({
        rerollTokens: 0,
        streakProtectionTokens: 2,
        ownedFrameIds: [],
        equippedFrameId: null,
        deepDiveLoreUnlocked: false,
        claimedMilestoneKeys: [],
      })
    );

    store.dispatch(
      recordActivity.fulfilled(
        { state: { currentStreak: 1, longestStreak: 4, lastActiveDate: '2026-01-05' }, protectionUsed: false },
        'test-request-id'
      )
    );

    expect(store.getState().rewards.streakProtectionTokens).toBe(2);
  });
});
