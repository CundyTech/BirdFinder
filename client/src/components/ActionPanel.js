import React from 'react';
import { View, TouchableOpacity, Text } from 'react-native';
import { useTranslation } from 'react-i18next';
import styles from '../styles';

export default function ActionPanel({ onPress }) {
  const { t } = useTranslation();
  return (
    <View style={styles.actionPanel}>
      <TouchableOpacity style={styles.mainActionButton} onPress={onPress}>
        <Text style={styles.mainActionText}>{t('components.actionPanel.takePhoto')}</Text>
      </TouchableOpacity>
    </View>
  );
}
