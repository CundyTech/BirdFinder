import AsyncStorage from '@react-native-async-storage/async-storage';
import { loadStreakState, recordActivity } from './streakStorage';

const STORAGE_KEY = 'streak.state.v1';

describe('streakStorage', () => {
  beforeEach(async () => {
    await AsyncStorage.clear();
  });

  describe('loadStreakState', () => {
    it('starts at zero with no history', async () => {
      expect(await loadStreakState()).toEqual({
        currentStreak: 0,
        longestStreak: 0,
        lastActiveDate: null,
      });
    });

    it('recovers from corrupted JSON instead of throwing', async () => {
      await AsyncStorage.setItem(STORAGE_KEY, 'not json{');
      expect(await loadStreakState()).toEqual({
        currentStreak: 0,
        longestStreak: 0,
        lastActiveDate: null,
      });
    });
  });

  describe('recordActivity', () => {
    it('starts the streak at 1 on first-ever activity', async () => {
      const { state, protectionUsed } = await recordActivity('2026-01-01', false);
      expect(state.currentStreak).toBe(1);
      expect(state.longestStreak).toBe(1);
      expect(state.lastActiveDate).toBe('2026-01-01');
      expect(protectionUsed).toBe(false);
    });

    it('does not double-count activity recorded twice on the same day', async () => {
      await recordActivity('2026-01-01', false);
      const { state } = await recordActivity('2026-01-01', false);
      expect(state.currentStreak).toBe(1);
    });

    it('increments the streak on a consecutive day', async () => {
      await recordActivity('2026-01-01', false);
      const { state } = await recordActivity('2026-01-02', false);
      expect(state.currentStreak).toBe(2);
      expect(state.longestStreak).toBe(2);
    });

    it('resets the streak after a gap with no protection available', async () => {
      await recordActivity('2026-01-01', false);
      await recordActivity('2026-01-02', false);
      const { state, protectionUsed } = await recordActivity('2026-01-05', false);
      expect(state.currentStreak).toBe(1);
      expect(state.longestStreak).toBe(2);
      expect(protectionUsed).toBe(false);
    });

    it('covers exactly one missed day when protection is available, and reports it was used', async () => {
      await recordActivity('2026-01-01', false);
      const { state, protectionUsed } = await recordActivity('2026-01-03', true);
      expect(protectionUsed).toBe(true);
      expect(state.currentStreak).toBe(2);
    });

    it('does not cover a gap of more than one missed day, even with protection available', async () => {
      await recordActivity('2026-01-01', false);
      const { state, protectionUsed } = await recordActivity('2026-01-10', true);
      expect(protectionUsed).toBe(false);
      expect(state.currentStreak).toBe(1);
    });

    it('keeps the highest streak ever reached as longestStreak after a later reset', async () => {
      await recordActivity('2026-01-01', false);
      await recordActivity('2026-01-02', false);
      await recordActivity('2026-01-03', false);
      const { state } = await recordActivity('2026-02-01', false);
      expect(state.currentStreak).toBe(1);
      expect(state.longestStreak).toBe(3);
    });
  });
});
