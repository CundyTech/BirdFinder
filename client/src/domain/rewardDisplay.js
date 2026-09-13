import i18n from '../i18n';
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
  if (reward.type === 'reroll') return i18n.t('rewards.names.freeReroll');
  if (reward.type === 'streakProtection') return i18n.t('rewards.names.streakProtection');
  if (reward.type === 'deepDiveLore') return i18n.t('rewards.names.deepDiveLore');
  return fallback;
}

// What the perk actually does once you have it, for display alongside
// rewardName once a milestone is unlocked.
export function rewardActionText(reward) {
  if (!reward) return null;
  if (reward.type === 'reroll') {
    return i18n.t('rewards.actionText.reroll', { count: reward.amount });
  }
  if (reward.type === 'streakProtection') {
    return i18n.t('rewards.actionText.streakProtection', { count: reward.amount });
  }
  if (reward.type === 'frame') return i18n.t('rewards.actionText.frame');
  if (reward.type === 'deepDiveLore') return i18n.t('rewards.actionText.deepDiveLore');
  return null;
}
