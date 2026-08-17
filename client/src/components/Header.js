import React from 'react';
import { View, Text, Image, ActivityIndicator, TouchableOpacity } from 'react-native';
import { useSelector } from 'react-redux';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import styles from '../styles';

export default function Header({ apiHealth, healthLoading, onRetryHealth }) {
  const isHealthy = apiHealth?.status === 'healthy';
  const isUnhealthy = apiHealth?.status === 'unhealthy';
  const filmBalance = useSelector((state) => state.film.balance);

  return (
    <View style={styles.header}>
      <View style={styles.brandRow}>
        <View style={styles.logoMark}>
          <Image source={require('../../assets/icon.png')} style={styles.logoMarkImage} />
        </View>
        <View style={styles.brandTextWrap}>
          <Text style={styles.brandTitle}>Bird Finder UK</Text>
          <Text style={styles.brandSubtitle}>UK bird identification</Text>
        </View>
        <View style={styles.filmBadge}>
          <MaterialCommunityIcons name="filmstrip" size={14} color={styles.PALETTE.accent} />
          <Text style={styles.filmBadgeText}>{filmBalance}</Text>
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
          <Text style={styles.healthBannerErrorText}>Can't reach the server, tap to retry</Text>
        </TouchableOpacity>
      )}
    </View>
  );
}
