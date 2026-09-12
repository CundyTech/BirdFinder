import React, { useState } from 'react';
import { Modal, View, Text, TouchableOpacity, ScrollView } from 'react-native';
import { MaterialCommunityIcons, Feather } from '@expo/vector-icons';
import { useSelector } from 'react-redux';
import styles from '../styles';
import useTrophyCategories from '../hooks/useTrophyCategories';
import { FRAMES } from '../domain/frames';
import { rewardDisplay, rewardName, rewardActionText } from '../domain/rewardDisplay';
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
  const [showFramePicker, setShowFramePicker] = useState(false);
  const categories = useTrophyCategories();
  const claimedKeys = useSelector((state) => state.rewards.claimedMilestoneKeys);
  const ownedFrameIds = useSelector((state) => state.rewards.ownedFrameIds);
  const milestones = categories.find((c) => c.id === 'milestones')?.trophies || [];
  const otherPerks = milestones.filter((t) => t.reward?.type !== 'frame');
  // Counts only ids that still match a known frame — ownedFrameIds can carry
  // a stale id from a frame that no longer exists (e.g. renamed during
  // development), which FramePickerModal already ignores by construction
  // (it iterates FRAMES, not ownedFrameIds) but a raw .length here wouldn't.
  const framesOwnedCount = ownedFrameIds.filter((id) => FRAMES.some((f) => f.id === id)).length;

  const close = () => {
    setShowFramePicker(false);
    onClose();
  };

  return (
    <>
      <Modal visible={visible} transparent animationType="fade" onRequestClose={close}>
        <View style={styles.filmModalBackdrop}>
          <View style={styles.filmModalCard}>
            <Text style={styles.filmModalTitle}>Perks</Text>
            <Text style={styles.filmModalSubtitle}>What you've earned from milestones.</Text>

            <ScrollView style={{ maxHeight: 420 }} showsVerticalScrollIndicator={false}>
              <TouchableOpacity
                style={styles.filmModalOption}
                onPress={() => setShowFramePicker(true)}
                activeOpacity={0.85}
                accessibilityRole="button"
                accessibilityLabel={`Frames, ${framesOwnedCount} of ${FRAMES.length} unlocked`}
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
                  <Text style={styles.filmModalOptionTitle}>Frames</Text>
                  <Text style={styles.filmModalOptionSub}>
                    {framesOwnedCount} of {FRAMES.length} unlocked — tap to choose which one to wear
                  </Text>
                </View>
                <Feather name="chevron-right" size={18} color={styles.PALETTE.mutedText} />
              </TouchableOpacity>

              {otherPerks.map((trophy) => {
                const unlocked = claimedKeys.includes(`milestones:${trophy.label}`);
                const display = rewardDisplay(trophy.reward);
                return (
                  <View
                    key={trophy.label}
                    style={[styles.filmModalOption, !unlocked && styles.filmModalOptionDisabled]}
                    accessibilityLabel={`${rewardName(trophy.reward, trophy.label)}, ${unlocked ? 'unlocked' : 'locked'}`}
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
            </ScrollView>

            <TouchableOpacity style={styles.filmModalClose} onPress={close}>
              <Text style={styles.filmModalCloseText}>Close</Text>
            </TouchableOpacity>
          </View>
        </View>
      </Modal>

      <FramePickerModal visible={showFramePicker} onClose={() => setShowFramePicker(false)} />
    </>
  );
}
