import AsyncStorage from '@react-native-async-storage/async-storage';
import {
  loadRewardsState,
  spendReroll,
  consumeStreakProtection,
  equipFrame,
  grantMilestoneRewards,
} from './rewardsStorage';

const STORAGE_KEY = 'rewards.state.v1';

function defaultState() {
  return {
    rerollTokens: 0,
    streakProtectionTokens: 0,
    ownedFrameIds: [],
    equippedFrameId: null,
    deepDiveLoreUnlocked: false,
    claimedMilestoneKeys: [],
  };
}

describe('rewardsStorage', () => {
  beforeEach(async () => {
    await AsyncStorage.clear();
  });

  describe('loadRewardsState', () => {
    it('starts empty with no history', async () => {
      expect(await loadRewardsState()).toEqual(defaultState());
    });

    it('recovers from corrupted JSON instead of throwing', async () => {
      await AsyncStorage.setItem(STORAGE_KEY, 'not json{');
      expect(await loadRewardsState()).toEqual(defaultState());
    });
  });

  describe('spendReroll', () => {
    it('returns null when there are no tokens to spend', async () => {
      expect(await spendReroll()).toBeNull();
    });

    it('decrements the token count', async () => {
      await grantMilestoneRewards([{ key: 'm1', reward: { type: 'reroll', amount: 2 } }]);
      const result = await spendReroll();
      expect(result.rerollTokens).toBe(1);
    });
  });

  describe('consumeStreakProtection', () => {
    it('returns null when there are no protection tokens', async () => {
      expect(await consumeStreakProtection()).toBeNull();
    });

    it('decrements the protection token count', async () => {
      await grantMilestoneRewards([{ key: 'm1', reward: { type: 'streakProtection', amount: 1 } }]);
      const result = await consumeStreakProtection();
      expect(result.streakProtectionTokens).toBe(0);
    });
  });

  describe('equipFrame', () => {
    it('does nothing if the frame is not owned', async () => {
      const result = await equipFrame('bronze-feather');
      expect(result.equippedFrameId).toBeNull();
    });

    it('equips a frame that is owned', async () => {
      await grantMilestoneRewards([{ key: 'm1', reward: { type: 'frame', frameId: 'bronze-feather' } }]);
      await grantMilestoneRewards([{ key: 'm2', reward: { type: 'frame', frameId: 'golden-wing' } }]);

      const result = await equipFrame('golden-wing');

      expect(result.equippedFrameId).toBe('golden-wing');
    });
  });

  describe('grantMilestoneRewards', () => {
    it('grants a reroll reward and records the claim', async () => {
      const { state, newlyClaimed } = await grantMilestoneRewards([
        { key: 'milestones:First Reroll', reward: { type: 'reroll', amount: 1 } },
      ]);
      expect(state.rerollTokens).toBe(1);
      expect(newlyClaimed).toEqual(['milestones:First Reroll']);
    });

    it('grants a frame reward, adds it to owned, and auto-equips the first frame ever earned', async () => {
      const { state } = await grantMilestoneRewards([
        { key: 'milestones:Frequent Flyer', reward: { type: 'frame', frameId: 'bronze-feather' } },
      ]);
      expect(state.ownedFrameIds).toEqual(['bronze-feather']);
      expect(state.equippedFrameId).toBe('bronze-feather');
    });

    it('does not change the equipped frame when a second frame is earned', async () => {
      await grantMilestoneRewards([
        { key: 'milestones:Frequent Flyer', reward: { type: 'frame', frameId: 'bronze-feather' } },
      ]);
      const { state } = await grantMilestoneRewards([
        { key: 'milestones:Dedicated Birder', reward: { type: 'frame', frameId: 'golden-wing' } },
      ]);
      expect(state.ownedFrameIds).toEqual(['bronze-feather', 'golden-wing']);
      expect(state.equippedFrameId).toBe('bronze-feather');
    });

    it('grants a deep-dive lore unlock', async () => {
      const { state } = await grantMilestoneRewards([
        { key: "milestones:Completionist's Notes", reward: { type: 'deepDiveLore' } },
      ]);
      expect(state.deepDiveLoreUnlocked).toBe(true);
    });

    it('does not re-grant an already-claimed milestone', async () => {
      await grantMilestoneRewards([{ key: 'm1', reward: { type: 'reroll', amount: 1 } }]);
      const { state, newlyClaimed } = await grantMilestoneRewards([
        { key: 'm1', reward: { type: 'reroll', amount: 1 } },
      ]);
      expect(state.rerollTokens).toBe(1);
      expect(newlyClaimed).toEqual([]);
    });

    it('grants only the newly-unlocked milestones when mixed with already-claimed ones', async () => {
      await grantMilestoneRewards([{ key: 'm1', reward: { type: 'reroll', amount: 1 } }]);
      const { state, newlyClaimed } = await grantMilestoneRewards([
        { key: 'm1', reward: { type: 'reroll', amount: 1 } },
        { key: 'm2', reward: { type: 'streakProtection', amount: 1 } },
      ]);
      expect(state.rerollTokens).toBe(1);
      expect(state.streakProtectionTokens).toBe(1);
      expect(newlyClaimed).toEqual(['m2']);
    });

    // Regression test: App.js's milestone-claim effect re-dispatches
    // grantMilestoneRewards on every trophy recompute (i.e. on every
    // sighting saved), so overlapping calls for the same still-unlocked
    // milestone are a real scenario, not a hypothetical. Without
    // serializing storage access, two overlapping calls can each read the
    // same pre-write state and both think the milestone is unclaimed,
    // granting the reroll token twice for one milestone.
    it('only grants a milestone once even when claimed by many overlapping calls', async () => {
      const milestone = [{ key: 'milestones:First Reroll', reward: { type: 'reroll', amount: 1 } }];

      await Promise.all([
        grantMilestoneRewards(milestone),
        grantMilestoneRewards(milestone),
        grantMilestoneRewards(milestone),
        grantMilestoneRewards(milestone),
        grantMilestoneRewards(milestone),
      ]);

      const final = await loadRewardsState();
      expect(final.rerollTokens).toBe(1);
      expect(final.claimedMilestoneKeys).toEqual(['milestones:First Reroll']);
    });
  });
});
