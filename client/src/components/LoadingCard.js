import React from 'react';
import { View, ActivityIndicator, Text } from 'react-native';
import styles from '../styles';

export default function LoadingCard() {
  return (
    <View style={styles.loadingCard}>
      <ActivityIndicator size="large" color={styles.PALETTE.primary} />
      <Text style={styles.loadingText}>Analyzing image...</Text>
      <Text style={styles.loadingSubtext}>This may take a few seconds</Text>
    </View>
  );
}
