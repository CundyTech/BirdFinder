import React from 'react';
import { View, ActivityIndicator, Text } from 'react-native';
import styles from '../styles';

export default function LoadingCard() {
  return (
    <View style={styles.loadingCard}>
      <View style={styles.loadingOrb}>
        <ActivityIndicator size="large" color={styles.PALETTE.primary} />
      </View>

      <Text style={styles.loadingEyebrow}>Bird recognition</Text>
      <Text style={styles.loadingText}>Scanning image...</Text>
      <Text style={styles.loadingSubtext}>Matching feather patterns</Text>
    </View>
  );
}
