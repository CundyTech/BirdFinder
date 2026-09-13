import React from 'react';
import { View, ActivityIndicator, Text } from 'react-native';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import { useTranslation } from 'react-i18next';
import styles from '../styles';

// spendKind mirrors HomeScreen's spend decision (an ad-earned free pass
// first, then Film, then a reroll token only once Film is at zero, nothing
// at all once unlocked forever) — shown here so the cost (or lack of one)
// isn't a surprise that only shows up as the header's Film count quietly
// changing after the fact.
const COST_BY_KIND = {
  film: { icon: 'filmstrip', color: 'accent', bg: 'rgba(245, 158, 11, 0.14)', textKey: 'components.loadingCard.usingFilm' },
  reroll: {
    icon: 'dice-multiple-outline',
    color: 'accent',
    bg: 'rgba(245, 158, 11, 0.14)',
    textKey: 'components.loadingCard.usingReroll',
  },
  'free-ad': {
    icon: 'play-circle-outline',
    color: 'primary',
    bg: 'rgba(31, 157, 107, 0.14)',
    textKey: 'components.loadingCard.freeViaAd',
  },
};

export default function LoadingCard({ spendKind = 'film' }) {
  const { t } = useTranslation();
  const cost = COST_BY_KIND[spendKind];

  return (
    <View style={styles.loadingCard}>
      <View style={styles.loadingOrb}>
        <ActivityIndicator size="large" color={styles.PALETTE.primary} />
      </View>

      <Text style={styles.loadingEyebrow}>{t('components.loadingCard.eyebrow')}</Text>
      <Text style={styles.loadingText}>{t('components.loadingCard.scanning')}</Text>
      <Text style={styles.loadingSubtext}>{t('components.loadingCard.matchingPatterns')}</Text>

      {cost && (
        <View style={[styles.loadingCostBadge, { backgroundColor: cost.bg }]}>
          <MaterialCommunityIcons name={cost.icon} size={14} color={styles.PALETTE[cost.color]} />
          <Text style={[styles.loadingCostBadgeText, { color: styles.PALETTE[cost.color] }]}>{t(cost.textKey)}</Text>
        </View>
      )}
    </View>
  );
}
