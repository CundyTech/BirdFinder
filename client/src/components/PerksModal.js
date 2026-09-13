import React, { useState } from 'react';
import { Modal, View, Text, TouchableOpacity, ScrollView, Alert, DevSettings } from 'react-native';
import { MaterialCommunityIcons, Feather } from '@expo/vector-icons';
import { useSelector, useDispatch } from 'react-redux';
import { useTranslation } from 'react-i18next';
import AsyncStorage from '@react-native-async-storage/async-storage';
import styles from '../styles';
import useTrophyCategories from '../hooks/useTrophyCategories';
import { FRAMES } from '../domain/frames';
import { rewardDisplay, rewardName, rewardActionText } from '../domain/rewardDisplay';
import { grantMilestoneRewards } from '../store/rewardsSlice';
import FramePickerModal from './FramePickerModal';

// Every milestone trophy (achievements.js) grants a perk other than Film,
// so this is scoped to just that category — species/rarity/etc trophies
// already have their own home in the Trophy Cabinet. Unlocked is read from
// claimedMilestoneKeys (the permanent grant record), not the trophy's own
// live-recomputed `unlocked` flag — sightings/streak progress can regress
// after a reward is earned (e.g. deleting old sightings), but a perk once
// granted stays owned regardless.
//
// Frame rewards are collapsed into one "Frames" row rather than listed per
// milestone — there's only one thing to actually do with a frame (equip
// it), so a single row popping FramePickerModal out on top (which shows
// every frame, locked ones greyed) is more useful than five near-identical
// "tap to choose which frame to wear" entries.
export default function PerksModal({ visible, onClose }) {
  const { t } = useTranslation();
  const dispatch = useDispatch();
  const [showFramePicker, setShowFramePicker] = useState(false);
  const categories = useTrophyCategories();
  const claimedKeys = useSelector((state) => state.rewards.claimedMilestoneKeys);
  const ownedFrameIds = useSelector((state) => state.rewards.ownedFrameIds);
  const milestones = categories.find((c) => c.id === 'milestones')?.trophies || [];
  const otherPerks = milestones.filter((trophy) => trophy.reward?.type !== 'frame');
  // Counts only ids that still match a known frame — ownedFrameIds can carry
  // a stale id from a frame that no longer exists (e.g. renamed during
  // development), which FramePickerModal already ignores by construction
  // (it iterates FRAMES, not ownedFrameIds) but a raw .length here wouldn't.
  const framesOwnedCount = ownedFrameIds.filter((id) => FRAMES.some((f) => f.id === id)).length;

  const close = () => {
    setShowFramePicker(false);
    onClose();
  };

  // Dev-only shortcuts for testing the reward flow without actually
  // grinding for it. milestones already carries every milestone's static
  // {label, reward} regardless of live progress, so no separate lookup is
  // needed to grant them all.
  const handleUnlockAllPerks = () => {
    const allMilestones = milestones.map((milestone) => ({ key: `milestones:${milestone.key}`, reward: milestone.reward }));
    dispatch(grantMilestoneRewards(allMilestones));
  };

  const handleWipeAllData = () => {
    Alert.alert(
      t('modals.perks.wipeConfirmTitle'),
      t('modals.perks.wipeConfirmMessage'),
      [
        { text: t('common.cancel'), style: 'cancel' },
        {
          text: t('modals.perks.wipe'),
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
    <>
      <Modal visible={visible} transparent animationType="fade" onRequestClose={close}>
        <View style={styles.filmModalBackdrop}>
          <View style={styles.filmModalCard}>
            <Text style={styles.filmModalTitle}>{t('modals.perks.title')}</Text>
            <Text style={styles.filmModalSubtitle}>{t('modals.perks.subtitle')}</Text>

            <ScrollView style={{ maxHeight: 420 }} showsVerticalScrollIndicator={false}>
              <TouchableOpacity
                style={styles.filmModalOption}
                onPress={() => setShowFramePicker(true)}
                activeOpacity={0.85}
                accessibilityRole="button"
                accessibilityLabel={t('modals.perks.framesAccessibilityLabel', { owned: framesOwnedCount, total: FRAMES.length })}
              >
                <View
                  style={[
                    styles.filmModalOptionIcon,
                    { backgroundColor: framesOwnedCount > 0 ? 'rgba(245, 158, 11, 0.16)' : styles.PALETTE.surface },
                  ]}
                >
                  <MaterialCommunityIcons
                    name="image-frame"
                    size={22}
                    color={framesOwnedCount > 0 ? styles.PALETTE.accent : styles.PALETTE.mutedText}
                  />
                </View>
                <View style={styles.filmModalOptionText}>
                  <Text style={styles.filmModalOptionTitle}>{t('modals.perks.framesTitle')}</Text>
                  <Text style={styles.filmModalOptionSub}>
                    {t('modals.perks.framesSub', { owned: framesOwnedCount, total: FRAMES.length })}
                  </Text>
                </View>
                <Feather name="chevron-right" size={18} color={styles.PALETTE.mutedText} />
              </TouchableOpacity>

              {otherPerks.map((trophy) => {
                const unlocked = claimedKeys.includes(`milestones:${trophy.key}`);
                const display = rewardDisplay(trophy.reward);
                return (
                  <View
                    key={trophy.key}
                    style={[styles.filmModalOption, !unlocked && styles.filmModalOptionDisabled]}
                    accessibilityLabel={`${rewardName(trophy.reward, trophy.label)}, ${unlocked ? t('common.unlocked') : t('common.locked')}`}
                  >
                    <View
                      style={[
                        styles.filmModalOptionIcon,
                        { backgroundColor: unlocked ? `${display?.color || styles.PALETTE.primary}29` : styles.PALETTE.surface },
                      ]}
                    >
                      <MaterialCommunityIcons
                        name={unlocked ? display?.icon || 'trophy' : 'lock'}
                        size={22}
                        color={unlocked ? display?.color || styles.PALETTE.primary : styles.PALETTE.mutedText}
                      />
                    </View>
                    <View style={styles.filmModalOptionText}>
                      <Text style={styles.filmModalOptionTitle}>{rewardName(trophy.reward, trophy.label)}</Text>
                      <Text style={styles.filmModalOptionSub}>
                        {unlocked ? rewardActionText(trophy.reward) : trophy.description}
                      </Text>
                    </View>
                  </View>
                );
              })}

              {__DEV__ && (
                <View>
                  <Text style={[styles.sectionLabel, { marginTop: 16 }]}>{t('modals.perks.debugSectionLabel')}</Text>
                  <TouchableOpacity style={styles.filmModalOption} onPress={handleUnlockAllPerks} activeOpacity={0.85}>
                    <View style={[styles.filmModalOptionIcon, { backgroundColor: 'rgba(245, 158, 11, 0.16)' }]}>
                      <MaterialCommunityIcons name="star-shooting-outline" size={22} color={styles.PALETTE.accent} />
                    </View>
                    <View style={styles.filmModalOptionText}>
                      <Text style={styles.filmModalOptionTitle}>{t('modals.perks.unlockAllPerksTitle')}</Text>
                      <Text style={styles.filmModalOptionSub}>{t('modals.perks.unlockAllPerksSub')}</Text>
                    </View>
                  </TouchableOpacity>
                  <TouchableOpacity style={styles.filmModalOption} onPress={handleWipeAllData} activeOpacity={0.85}>
                    <View style={[styles.filmModalOptionIcon, { backgroundColor: 'rgba(244, 67, 54, 0.16)' }]}>
                      <MaterialCommunityIcons name="delete-outline" size={22} color={styles.PALETTE.danger} />
                    </View>
                    <View style={styles.filmModalOptionText}>
                      <Text style={styles.filmModalOptionTitle}>{t('modals.perks.wipeAllDataTitle')}</Text>
                      <Text style={styles.filmModalOptionSub}>{t('modals.perks.wipeAllDataSub')}</Text>
                    </View>
                  </TouchableOpacity>
                </View>
              )}
            </ScrollView>

            <TouchableOpacity style={styles.filmModalClose} onPress={close}>
              <Text style={styles.filmModalCloseText}>{t('common.close')}</Text>
            </TouchableOpacity>
          </View>
        </View>
      </Modal>

      <FramePickerModal visible={showFramePicker} onClose={() => setShowFramePicker(false)} />
    </>
  );
}
