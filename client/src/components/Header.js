import React, { useState } from 'react';
import { View, Text, Image, ActivityIndicator, TouchableOpacity } from 'react-native';
import { useSelector } from 'react-redux';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import styles from '../styles';
import { getFrame } from '../domain/frames';
import FramePickerModal from './FramePickerModal';

export default function Header({ apiHealth, healthLoading, onRetryHealth }) {
  const isHealthy = apiHealth?.status === 'healthy';
  const isUnhealthy = apiHealth?.status === 'unhealthy';
  const filmBalance = useSelector((state) => state.film.balance);
  const unlockedForever = useSelector((state) => state.premium.unlockedForever);
  const currentStreak = useSelector((state) => state.streak.currentStreak);
  const ownedFrameIds = useSelector((state) => state.rewards.ownedFrameIds);
  const equippedFrameId = useSelector((state) => state.rewards.equippedFrameId);
  const [showFramePicker, setShowFramePicker] = useState(false);

  const equippedFrame = getFrame(equippedFrameId);

  return (
    <View style={styles.header}>
      <View style={styles.brandRow}>
        <TouchableOpacity
          onPress={() => setShowFramePicker(true)}
          disabled={ownedFrameIds.length === 0}
          activeOpacity={0.8}
          accessibilityRole="button"
          accessibilityLabel={equippedFrame ? `${equippedFrame.label} frame, tap to change` : 'App icon'}
        >
          <View
            style={[
              styles.logoMarkFrame,
              equippedFrame && { borderColor: equippedFrame.ringColor, backgroundColor: `${equippedFrame.ringColor}33` },
            ]}
          >
            <View style={styles.logoMark}>
              <Image source={require('../../assets/icon.png')} style={styles.logoMarkImage} />
            </View>
            {equippedFrame && (
              <View style={[styles.logoMarkBadge, { backgroundColor: equippedFrame.ringColor }]}>
                <MaterialCommunityIcons name={equippedFrame.icon} size={11} color="#0e1116" />
              </View>
            )}
          </View>
        </TouchableOpacity>
        <View style={styles.brandTextWrap}>
          <Text style={styles.brandTitle}>Bird Finder UK</Text>
          <Text style={styles.brandSubtitle}>UK bird identification</Text>
        </View>
        {currentStreak > 0 && (
          <View style={styles.streakBadge}>
            <MaterialCommunityIcons name="fire" size={14} color={styles.PALETTE.danger} />
            <Text style={styles.streakBadgeText}>{currentStreak}</Text>
          </View>
        )}
        <View style={styles.filmBadge}>
          <MaterialCommunityIcons name="filmstrip" size={14} color={styles.PALETTE.accent} />
          {unlockedForever ? (
            <MaterialCommunityIcons name="infinity" size={14} color={styles.PALETTE.accent} />
          ) : (
            <Text style={styles.filmBadgeText}>{filmBalance}</Text>
          )}
        </View>
        {!healthLoading && apiHealth && (
          <View style={[styles.statusIndicator, isHealthy ? styles.statusHealthy : styles.statusUnhealthy]} />
        )}
      </View>

      <FramePickerModal visible={showFramePicker} onClose={() => setShowFramePicker(false)} />

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
