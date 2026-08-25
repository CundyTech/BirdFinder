import AsyncStorage from '@react-native-async-storage/async-storage';
import { createSemaphore } from './asyncSemaphore';

const STORAGE_KEY = 'rewards.state.v1';

// Every exported mutator here does read-modify-write against the same
// storage key. App.js's milestone-claim effect re-dispatches
// grantMilestoneRewards on every trophy recompute (i.e. on every sighting),
// and HomeScreen can spend a reroll around the same time — without this,
// two overlapping calls can each read the same pre-write state and one
// update clobbers the other (e.g. a milestone reward silently lost, or
// worse, re-applied on a later render because its claim never actually
// persisted). A semaphore of 1 serializes them into a queue instead.
const withLock = createSemaphore(1);

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

function sanitize(parsed) {
  if (!parsed || typeof parsed !== 'object') return defaultState();
  return {
    rerollTokens: typeof parsed.rerollTokens === 'number' ? parsed.rerollTokens : 0,
    streakProtectionTokens: typeof parsed.streakProtectionTokens === 'number' ? parsed.streakProtectionTokens : 0,
    ownedFrameIds: Array.isArray(parsed.ownedFrameIds) ? parsed.ownedFrameIds : [],
    equippedFrameId: typeof parsed.equippedFrameId === 'string' ? parsed.equippedFrameId : null,
    deepDiveLoreUnlocked: Boolean(parsed.deepDiveLoreUnlocked),
    claimedMilestoneKeys: Array.isArray(parsed.claimedMilestoneKeys) ? parsed.claimedMilestoneKeys : [],
  };
}

export async function loadRewardsState() {
  const raw = await AsyncStorage.getItem(STORAGE_KEY);
  if (!raw) return defaultState();
  try {
    return sanitize(JSON.parse(raw));
  } catch {
    return defaultState();
  }
}

async function saveRewardsState(state) {
  await AsyncStorage.setItem(STORAGE_KEY, JSON.stringify(state));
}

// Spends one reroll token. Returns the updated state, or null if none were
// available — mirrors filmStorage.spendFilm's null-means-nothing-to-spend
// convention.
export function spendReroll() {
  return withLock(async () => {
    const state = await loadRewardsState();
    if (state.rerollTokens <= 0) return null;
    const updated = { ...state, rerollTokens: state.rerollTokens - 1 };
    await saveRewardsState(updated);
    return updated;
  });
}

// Consumed by streakStorage.recordActivity when a missed day is covered.
// Returns the updated state, or null if there was nothing to spend (the
// caller should treat that as "couldn't protect the streak").
export function consumeStreakProtection() {
  return withLock(async () => {
    const state = await loadRewardsState();
    if (state.streakProtectionTokens <= 0) return null;
    const updated = { ...state, streakProtectionTokens: state.streakProtectionTokens - 1 };
    await saveRewardsState(updated);
    return updated;
  });
}

export function equipFrame(frameId) {
  return withLock(async () => {
    const state = await loadRewardsState();
    if (!state.ownedFrameIds.includes(frameId)) return state;
    const updated = { ...state, equippedFrameId: frameId };
    await saveRewardsState(updated);
    return updated;
  });
}

// Applies whichever milestones in `milestones` (each { key, reward }) haven't
// already been claimed. Callers can pass every currently-unlocked milestone
// every time — already-claimed ones are filtered out here, matching
// filmStorage.claimTrophyRewards' idempotent-by-diffing pattern.
export function grantMilestoneRewards(milestones) {
  return withLock(async () => {
    const state = await loadRewardsState();
    const newOnes = milestones.filter((m) => !state.claimedMilestoneKeys.includes(m.key));
    if (newOnes.length === 0) return { state, newlyClaimed: [] };

    const updated = {
      ...state,
      claimedMilestoneKeys: [...state.claimedMilestoneKeys, ...newOnes.map((m) => m.key)],
    };

    for (const { reward } of newOnes) {
      if (reward.type === 'reroll') {
        updated.rerollTokens += reward.amount;
      } else if (reward.type === 'streakProtection') {
        updated.streakProtectionTokens += reward.amount;
      } else if (reward.type === 'frame') {
        if (!updated.ownedFrameIds.includes(reward.frameId)) {
          updated.ownedFrameIds = [...updated.ownedFrameIds, reward.frameId];
        }
        // Auto-equip the first frame ever earned so the reward is visible
        // without the player needing to discover the picker.
        if (!updated.equippedFrameId) {
          updated.equippedFrameId = reward.frameId;
        }
      } else if (reward.type === 'deepDiveLore') {
        updated.deepDiveLoreUnlocked = true;
      }
    }

    await saveRewardsState(updated);
    return { state: updated, newlyClaimed: newOnes.map((m) => m.key) };
  });
}
