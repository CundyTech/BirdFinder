import React from 'react';
import { View, Text } from 'react-native';
import styles from '../styles';
import { RARITY_MAX_PIPS } from '../rarity';

// Shared between the full species facts card and the compact life-list
// tile — `compact` swaps in smaller pips and drops the "UK rarity" label
// (tile space is a fraction of the card's).
export default function RarityMeter({ rarity, compact }) {
  if (!rarity) return null;

  return (
    <View style={compact ? styles.tileRarityRow : styles.rarityRow}>
      {!compact && <Text style={styles.rarityLabel}>UK rarity</Text>}
      <View style={compact ? styles.tileRarityPipsRow : styles.rarityPipsRow}>
        {Array.from({ length: RARITY_MAX_PIPS }).map((_, i) => (
          <View
            key={i}
            style={[
              compact ? styles.tileRarityPip : styles.rarityPip,
              i < rarity.pips && (compact ? styles.tileRarityPipFilled : styles.rarityPipFilled),
            ]}
          />
        ))}
      </View>
      <Text style={compact ? styles.tileRarityLabel : styles.rarityTierText} numberOfLines={1}>
        {rarity.label}
      </Text>
    </View>
  );
}
