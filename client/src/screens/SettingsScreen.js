import React from 'react';
import { SafeAreaView, ScrollView, View, Text, TouchableOpacity, Alert, DevSettings } from 'react-native';
import { Feather } from '@expo/vector-icons';
import { useDispatch, useSelector } from 'react-redux';
import { useTranslation } from 'react-i18next';
import AsyncStorage from '@react-native-async-storage/async-storage';
import useAppStyles from '../hooks/useAppStyles';
import { setTheme } from '../store/settingsSlice';
import { grantMilestoneRewards } from '../store/rewardsSlice';
import useTrophyCategories from '../hooks/useTrophyCategories';

const THEME_OPTIONS = [
  { value: 'system', icon: 'smartphone', labelKey: 'settings.appearance.system', subKey: 'settings.appearance.systemSub' },
  { value: 'light', icon: 'sun', labelKey: 'settings.appearance.light' },
  { value: 'dark', icon: 'moon', labelKey: 'settings.appearance.dark' },
];

export default function SettingsScreen({ onBack }) {
  const { t } = useTranslation();
  const styles = useAppStyles();
  const dispatch = useDispatch();
  const theme = useSelector((state) => state.settings.theme);
  const categories = useTrophyCategories();

  // Dev-only shortcut for testing the reward flow without actually grinding
  // for it. milestones already carries every milestone's static {key,
  // label, reward} regardless of live progress, so no separate lookup is
  // needed to grant them all.
  const handleUnlockAllPerks = () => {
    const milestones = categories.find((c) => c.id === 'milestones')?.trophies || [];
    const allMilestones = milestones.map((milestone) => ({ key: `milestones:${milestone.key}`, reward: milestone.reward }));
    dispatch(grantMilestoneRewards(allMilestones));
  };

  const handleWipeAllData = () => {
    Alert.alert(
      t('settings.debug.wipeConfirmTitle'),
      t('settings.debug.wipeConfirmMessage'),
      [
        { text: t('common.cancel'), style: 'cancel' },
        {
          text: t('settings.debug.wipe'),
          style: 'destructive',
          onPress: async () => {
            await AsyncStorage.clear();
            DevSettings.reload();
          },
        },
      ]
    );
  };

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.subHeader}>
        <TouchableOpacity
          style={styles.subHeaderBackButton}
          onPress={onBack}
          accessibilityRole="button"
          accessibilityLabel={t('common.back')}
        >
          <Feather name="chevron-left" size={22} color={styles.PALETTE.textOnDark} />
        </TouchableOpacity>
        <View style={styles.subHeaderTitleWrap}>
          <Text style={styles.subHeaderTitle}>{t('settings.title')}</Text>
        </View>
      </View>

      <ScrollView
        contentContainerStyle={styles.scrollContainer}
        showsVerticalScrollIndicator={false}
        style={styles.mainContent}
      >
        <View style={styles.settingsSection}>
          <Text style={styles.sectionLabel}>{t('settings.appearance.sectionLabel')}</Text>
          {THEME_OPTIONS.map((option) => {
            const selected = theme === option.value;
            return (
              <TouchableOpacity
                key={option.value}
                style={[styles.settingsRow, selected && styles.settingsRowSelected]}
                onPress={() => dispatch(setTheme(option.value))}
                activeOpacity={0.85}
                accessibilityRole="radio"
                accessibilityState={{ selected }}
              >
                <View style={styles.filmModalOptionIcon}>
                  <Feather name={option.icon} size={20} color={selected ? styles.PALETTE.primary : styles.PALETTE.mutedText} />
                </View>
                <View style={styles.settingsRowTextWrap}>
                  <Text style={styles.settingsRowTitle}>{t(option.labelKey)}</Text>
                  {option.subKey && <Text style={styles.settingsRowSub}>{t(option.subKey)}</Text>}
                </View>
                {selected && <Feather name="check-circle" size={20} color={styles.PALETTE.primary} />}
              </TouchableOpacity>
            );
          })}
        </View>

        {__DEV__ && (
          <View style={styles.settingsSection}>
            <Text style={styles.sectionLabel}>{t('settings.debug.sectionLabel')}</Text>
            <TouchableOpacity style={styles.filmModalOption} onPress={handleUnlockAllPerks} activeOpacity={0.85}>
              <View style={[styles.filmModalOptionIcon, { backgroundColor: 'rgba(245, 158, 11, 0.16)' }]}>
                <Feather name="star" size={20} color={styles.PALETTE.accent} />
              </View>
              <View style={styles.filmModalOptionText}>
                <Text style={styles.filmModalOptionTitle}>{t('settings.debug.unlockAllPerksTitle')}</Text>
                <Text style={styles.filmModalOptionSub}>{t('settings.debug.unlockAllPerksSub')}</Text>
              </View>
            </TouchableOpacity>
            <TouchableOpacity style={styles.filmModalOption} onPress={handleWipeAllData} activeOpacity={0.85}>
              <View style={[styles.filmModalOptionIcon, { backgroundColor: 'rgba(244, 67, 54, 0.16)' }]}>
                <Feather name="trash-2" size={20} color={styles.PALETTE.danger} />
              </View>
              <View style={styles.filmModalOptionText}>
                <Text style={styles.filmModalOptionTitle}>{t('settings.debug.wipeAllDataTitle')}</Text>
                <Text style={styles.filmModalOptionSub}>{t('settings.debug.wipeAllDataSub')}</Text>
              </View>
            </TouchableOpacity>
          </View>
        )}
      </ScrollView>
    </SafeAreaView>
  );
}
