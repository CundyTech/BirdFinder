import React, { useState } from 'react';
import { View, Text, Image, TouchableOpacity } from 'react-native';
import { useSelector } from 'react-redux';
import { MaterialCommunityIcons, Feather } from '@expo/vector-icons';
import { useTranslation } from 'react-i18next';
import useAppStyles from '../hooks/useAppStyles';
import { getFrame } from '../domain/frames';
import PerksModal from './PerksModal';

export default function Header({ onOpenSettings }) {
  const { t } = useTranslation();
  const styles = useAppStyles();
  const filmBalance = useSelector((state) => state.film.balance);
  const unlockedForever = useSelector((state) => state.premium.unlockedForever);
  const currentStreak = useSelector((state) => state.streak.currentStreak);
  const equippedFrameId = useSelector((state) => state.rewards.equippedFrameId);
  const [showPerks, setShowPerks] = useState(false);

  const equippedFrame = getFrame(equippedFrameId);

  return (
    <View style={styles.header}>
      <View style={styles.brandRow}>
        <TouchableOpacity
          onPress={() => setShowPerks(true)}
          activeOpacity={0.8}
          accessibilityRole="button"
          accessibilityLabel={
            equippedFrame
              ? t('components.header.frameAccessibilityLabel', { frameLabel: equippedFrame.label })
              : t('components.header.viewPerks')
          }
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
          <Text style={styles.brandTitle}>{t('components.header.brandTitle')}</Text>
          <Text style={styles.brandSubtitle}>{t('components.header.brandSubtitle')}</Text>
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
        <TouchableOpacity
          onPress={onOpenSettings}
          activeOpacity={0.7}
          accessibilityRole="button"
          accessibilityLabel={t('components.header.openSettings')}
          style={styles.settingsButton}
        >
          <Feather name="settings" size={20} color={styles.PALETTE.mutedText} />
        </TouchableOpacity>
      </View>

      <PerksModal visible={showPerks} onClose={() => setShowPerks(false)} />
    </View>
  );
}
