import styles from '../styles';
import { getFrame } from './frames';

// Maps a milestone trophy's reward spec (see achievements.js) to a small
// icon/colour — used anywhere a reward needs to be shown without knowing
// its type up front (TrophyCard's badge, AchievementsModal's row icon).
export function rewardDisplay(reward) {
  if (!reward) return null;
  if (reward.type === 'reroll') {
    return { icon: 'dice-multiple-outline', color: styles.PALETTE.primary };
  }
  if (reward.type === 'streakProtection') {
    return { icon: 'shield-check-outline', color: styles.PALETTE.primary };
  }
  if (reward.type === 'frame') {
    const frame = getFrame(reward.frameId);
    return { icon: frame?.icon || 'star-four-points', color: frame?.ringColor || styles.PALETTE.accent };
  }
  if (reward.type === 'deepDiveLore') {
    return { icon: 'book-open-page-variant', color: styles.PALETTE.primary };
  }
  return null;
}

// The perk's own name — what it grants, not the milestone that grants it
// (e.g. "Floral Frame", not "Frequent Flyer") — for surfaces oriented
// around the reward itself rather than the achievement (AchievementsModal).
export function rewardName(reward, fallback) {
  if (!reward) return fallback;
  if (reward.type === 'frame') return getFrame(reward.frameId)?.label || fallback;
  if (reward.type === 'reroll') return 'Free Reroll';
  if (reward.type === 'streakProtection') return 'Streak Protection';
  if (reward.type === 'deepDiveLore') return 'Deep-Dive Lore';
  return fallback;
}

// What the perk actually does once you have it, for display alongside
// rewardName once a milestone is unlocked.
export function rewardActionText(reward) {
  if (!reward) return null;
  if (reward.type === 'reroll') {
    return `+${reward.amount} token${reward.amount === 1 ? '' : 's'} — skips the Film cost once each`;
  }
  if (reward.type === 'streakProtection') {
    return `+${reward.amount} token${reward.amount === 1 ? '' : 's'} — covers one missed day each`;
  }
  if (reward.type === 'frame') return 'Tap to choose which frame to wear';
  if (reward.type === 'deepDiveLore') return 'Extra lore unlocked on every species profile';
  return null;
}
