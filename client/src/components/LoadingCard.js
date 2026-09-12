import React from 'react';
import { View, ActivityIndicator, Text } from 'react-native';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import styles from '../styles';

// spendKind mirrors HomeScreen's spend decision (an ad-earned free pass
// first, then Film, then a reroll token only once Film is at zero, nothing
// at all once unlocked forever) — shown here so the cost (or lack of one)
// isn't a surprise that only shows up as the header's Film count quietly
// changing after the fact.
const COST_BY_KIND = {
  film: { icon: 'filmstrip', color: 'accent', bg: 'rgba(245, 158, 11, 0.14)', text: 'Using 1 Film for this ID' },
  reroll: {
    icon: 'dice-multiple-outline',
    color: 'accent',
    bg: 'rgba(245, 158, 11, 0.14)',
    text: 'Using 1 reroll token for this ID',
  },
  'free-ad': {
    icon: 'play-circle-outline',
    color: 'primary',
    bg: 'rgba(31, 157, 107, 0.14)',
    text: 'Free — covered by your ad',
  },
};

export default function LoadingCard({ spendKind = 'film' }) {
  const cost = COST_BY_KIND[spendKind];

  return (
    <View style={styles.loadingCard}>
      <View style={styles.loadingOrb}>
        <ActivityIndicator size="large" color={styles.PALETTE.primary} />
      </View>

      <Text style={styles.loadingEyebrow}>Bird recognition</Text>
      <Text style={styles.loadingText}>Scanning image...</Text>
      <Text style={styles.loadingSubtext}>Matching feather patterns</Text>

      {cost && (
        <View style={[styles.loadingCostBadge, { backgroundColor: cost.bg }]}>
          <MaterialCommunityIcons name={cost.icon} size={14} color={styles.PALETTE[cost.color]} />
          <Text style={[styles.loadingCostBadgeText, { color: styles.PALETTE[cost.color] }]}>{cost.text}</Text>
        </View>
      )}
    </View>
  );
}
