import React from 'react';
import { View, TouchableOpacity, Text } from 'react-native';
import styles from '../styles';

export default function ActionPanel({ onPress }) {
  return (
    <View style={styles.actionPanel}>
      <TouchableOpacity style={styles.mainActionButton} onPress={onPress}>
        <Text style={styles.mainActionText}>Take Photo</Text>
      </TouchableOpacity>
    </View>
  );
}
