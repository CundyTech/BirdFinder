import AsyncStorage from '@react-native-async-storage/async-storage';
import {
  loadFilmState,
  spendFilm,
  claimDailyRefill,
  claimTrophyRewards,
  grantAdReward,
  STARTING_BALANCE,
} from './filmStorage';

const STORAGE_KEY = 'film.state.v1';

describe('filmStorage', () => {
  beforeEach(async () => {
    await AsyncStorage.clear();
  });

  describe('loadFilmState', () => {
    it('returns the starting balance and no history on first-ever load', async () => {
      expect(await loadFilmState()).toEqual({
        balance: STARTING_BALANCE,
        lastRefillDate: null,
        claimedTrophyKeys: [],
        adsWatchedToday: 0,
        adsWatchedDate: null,
      });
    });

    it('recovers from corrupted JSON instead of throwing', async () => {
      await AsyncStorage.setItem(STORAGE_KEY, 'not json{');
      expect(await loadFilmState()).toEqual({
        balance: STARTING_BALANCE,
        lastRefillDate: null,
        claimedTrophyKeys: [],
        adsWatchedToday: 0,
        adsWatchedDate: null,
      });
    });

    it('sanitizes a partially-shaped stored value instead of trusting it blindly', async () => {
      await AsyncStorage.setItem(STORAGE_KEY, JSON.stringify({ balance: 3 }));
      expect(await loadFilmState()).toEqual({
        balance: 3,
        lastRefillDate: null,
        claimedTrophyKeys: [],
        adsWatchedToday: 0,
        adsWatchedDate: null,
      });
    });
  });

  describe('spendFilm', () => {
    it('decrements the balance by one', async () => {
      const updated = await spendFilm();
      expect(updated.balance).toBe(STARTING_BALANCE - 1);
    });

    it('persists the decrement', async () => {
      await spendFilm();
      expect((await loadFilmState()).balance).toBe(STARTING_BALANCE - 1);
    });

    it('returns null instead of going negative when the balance is already zero', async () => {
      for (let i = 0; i < STARTING_BALANCE; i += 1) {
        await spendFilm(); // eslint-disable-line no-await-in-loop
      }
      expect((await loadFilmState()).balance).toBe(0);

      const result = await spendFilm();

      expect(result).toBeNull();
      expect((await loadFilmState()).balance).toBe(0);
    });
  });

  describe('claimDailyRefill', () => {
    it('grants the refill amount and records the date', async () => {
      const updated = await claimDailyRefill(2, '2026-01-01');
      expect(updated.balance).toBe(STARTING_BALANCE + 2);
      expect(updated.lastRefillDate).toBe('2026-01-01');
    });

    it('does not grant a second refill for the same day', async () => {
      await claimDailyRefill(2, '2026-01-01');
      const secondAttempt = await claimDailyRefill(2, '2026-01-01');

      expect(secondAttempt.balance).toBe(STARTING_BALANCE + 2);
    });

    it('grants again on a new day', async () => {
      await claimDailyRefill(2, '2026-01-01');
      const nextDay = await claimDailyRefill(2, '2026-01-02');

      expect(nextDay.balance).toBe(STARTING_BALANCE + 4);
      expect(nextDay.lastRefillDate).toBe('2026-01-02');
    });
  });

  describe('claimTrophyRewards', () => {
    it('grants the reward for each newly unlocked trophy', async () => {
      const { state, newlyClaimed } = await claimTrophyRewards(10, ['rarity:Very rare', 'types:Tits']);

      expect(state.balance).toBe(STARTING_BALANCE + 20);
      expect(state.claimedTrophyKeys).toEqual(['rarity:Very rare', 'types:Tits']);
      expect(newlyClaimed).toEqual(['rarity:Very rare', 'types:Tits']);
    });

    it('does not re-grant a trophy that was already claimed', async () => {
      await claimTrophyRewards(10, ['rarity:Very rare']);
      const { state, newlyClaimed } = await claimTrophyRewards(10, ['rarity:Very rare']);

      expect(state.balance).toBe(STARTING_BALANCE + 10);
      expect(newlyClaimed).toEqual([]);
    });

    it('grants only the new keys when mixed with already-claimed ones', async () => {
      await claimTrophyRewards(10, ['rarity:Very rare']);
      const { state, newlyClaimed } = await claimTrophyRewards(10, ['rarity:Very rare', 'types:Tits']);

      expect(state.balance).toBe(STARTING_BALANCE + 20);
      expect(newlyClaimed).toEqual(['types:Tits']);
    });
  });

  describe('grantAdReward', () => {
    it('grants the reward and counts as one watch for the day', async () => {
      const { state, granted } = await grantAdReward(5, 3, '2026-01-01');

      expect(granted).toBe(true);
      expect(state.balance).toBe(STARTING_BALANCE + 5);
      expect(state.adsWatchedToday).toBe(1);
      expect(state.adsWatchedDate).toBe('2026-01-01');
    });

    it('stops granting once the daily cap is reached', async () => {
      await grantAdReward(5, 2, '2026-01-01');
      await grantAdReward(5, 2, '2026-01-01');
      const third = await grantAdReward(5, 2, '2026-01-01');

      expect(third.granted).toBe(false);
      expect(third.state.balance).toBe(STARTING_BALANCE + 10);
    });

    it('resets the watch count on a new day', async () => {
      await grantAdReward(5, 1, '2026-01-01');
      const nextDay = await grantAdReward(5, 1, '2026-01-02');

      expect(nextDay.granted).toBe(true);
      expect(nextDay.state.adsWatchedToday).toBe(1);
      expect(nextDay.state.adsWatchedDate).toBe('2026-01-02');
      expect(nextDay.state.balance).toBe(STARTING_BALANCE + 10);
    });
  });
});
