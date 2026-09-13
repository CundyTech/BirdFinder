import React from 'react';
import { View, Text } from 'react-native';
import { useTranslation } from 'react-i18next';
import useAppStyles from '../hooks/useAppStyles';

export default function PlaceholderCard() {
  const { t } = useTranslation();
  const styles = useAppStyles();
  return (
    <View style={styles.placeholderContainer}>
      <View style={styles.placeholderCard}>
        <Text style={{ fontSize: 20, color: styles.PALETTE.bg, fontWeight: '700' }}>{t('components.placeholderCard.title')}</Text>
        <Text style={{ marginTop: 8, color: styles.PALETTE.mutedText }}>{t('components.placeholderCard.subtitle')}</Text>
      </View>
    </View>
  );
}
