import React from 'react';
import { View, Text } from 'react-native';
import Svg, { Path, Circle } from 'react-native-svg';
import styles from '../styles';

// Deliberately schematic, not a real projected map — just a curved path
// between two labeled points. Keeps this on react-native-svg (already a
// dependency) instead of a native maps library, which would need a custom
// dev client and break Expo Go.
export default function MigrationMap({ breeding, wintering }) {
  return (
    <View style={styles.migrationMapCard}>
      <Svg width="100%" height={90} viewBox="0 0 300 90">
        <Path
          d="M36,18 C120,18 180,72 264,72"
          stroke={styles.PALETTE.primary}
          strokeWidth={2}
          strokeDasharray="5,6"
          fill="none"
        />
        <Circle cx={36} cy={18} r={6} fill={styles.PALETTE.primary} />
        <Circle cx={264} cy={72} r={6} fill={styles.PALETTE.accent} />
      </Svg>
      <View style={styles.migrationLabelRow}>
        <View style={styles.migrationLabelItem}>
          <View style={[styles.migrationDot, { backgroundColor: styles.PALETTE.primary }]} />
          <View>
            <Text style={styles.migrationLabelCaption}>Breeds</Text>
            <Text style={styles.migrationLabelText}>{breeding}</Text>
          </View>
        </View>
        <View style={styles.migrationLabelItem}>
          <View style={[styles.migrationDot, { backgroundColor: styles.PALETTE.accent }]} />
          <View>
            <Text style={styles.migrationLabelCaption}>Winters</Text>
            <Text style={styles.migrationLabelText}>{wintering}</Text>
          </View>
        </View>
      </View>
    </View>
  );
}
