import React from 'react';
import { View, Text } from 'react-native';
import styles from '../styles';

export default function PlaceholderCard() {
  return (
    <View style={styles.placeholderContainer}>
      <View style={styles.placeholderCard}>
        <Text style={{ fontSize: 20, color: styles.PALETTE.bg, fontWeight: '700' }}>Ready to Identify Birds?</Text>
        <Text style={{ marginTop: 8, color: styles.PALETTE.mutedText }}>Take a photo and our model will identify the species. Works best with clear photos.</Text>
      </View>
    </View>
  );
}
