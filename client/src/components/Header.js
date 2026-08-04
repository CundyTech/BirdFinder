import React from 'react';
import { View, Text, ActivityIndicator, TouchableOpacity } from 'react-native';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import styles from '../styles';

export default function Header({ apiHealth, healthLoading, onRetryHealth }) {
  const isHealthy = apiHealth?.status === 'healthy';
  const isUnhealthy = apiHealth?.status === 'unhealthy';

  return (
    <View style={styles.header}>
      <View style={styles.brandRow}>
        <View style={styles.logoMark}>
          <MaterialCommunityIcons name="bird" size={22} color="#ffffff" />
        </View>
        <View style={styles.brandTextWrap}>
          <Text style={styles.brandTitle}>BirdFinder</Text>
          <Text style={styles.brandSubtitle}>UK bird identification</Text>
        </View>
        {!healthLoading && apiHealth && (
          <View style={[styles.statusIndicator, isHealthy ? styles.statusHealthy : styles.statusUnhealthy]} />
        )}
      </View>

      {healthLoading && (
        <View style={styles.healthBanner}>
          <ActivityIndicator size="small" color={styles.PALETTE.mutedText} />
          <Text style={styles.healthBannerText}>Checking connection...</Text>
        </View>
      )}

      {!healthLoading && isUnhealthy && (
        <TouchableOpacity style={styles.healthBannerError} onPress={onRetryHealth}>
          <Text style={styles.healthBannerErrorText}>Can't reach the server — tap to retry</Text>
        </TouchableOpacity>
      )}
    </View>
  );
}
