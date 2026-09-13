import React from 'react';
import { Modal, View, Text, TouchableOpacity } from 'react-native';
import { MaterialCommunityIcons, Feather } from '@expo/vector-icons';
import { useDispatch, useSelector } from 'react-redux';
import { useTranslation } from 'react-i18next';
import styles from '../styles';
import { FRAMES } from '../domain/frames';
import { equipFrame } from '../store/rewardsSlice';

export default function FramePickerModal({ visible, onClose }) {
  const { t } = useTranslation();
  const dispatch = useDispatch();
  const ownedFrameIds = useSelector((state) => state.rewards.ownedFrameIds);
  const equippedFrameId = useSelector((state) => state.rewards.equippedFrameId);

  return (
    <Modal visible={visible} transparent animationType="fade" onRequestClose={onClose}>
      <View style={styles.filmModalBackdrop}>
        <View style={styles.filmModalCard}>
          <Text style={styles.filmModalTitle}>{t('modals.framePicker.title')}</Text>
          <Text style={styles.filmModalSubtitle}>{t('modals.framePicker.subtitle')}</Text>

          {FRAMES.map((frame) => {
            const owned = ownedFrameIds.includes(frame.id);
            const selected = frame.id === equippedFrameId;
            return (
              <TouchableOpacity
                key={frame.id}
                style={[styles.filmModalOption, !owned && styles.filmModalOptionDisabled]}
                onPress={() => owned && dispatch(equipFrame(frame.id))}
                disabled={!owned}
                activeOpacity={owned ? 0.85 : 1}
              >
                <View
                  style={[
                    styles.filmModalOptionIcon,
                    { backgroundColor: owned ? `${frame.ringColor}29` : styles.PALETTE.surface },
                  ]}
                >
                  <MaterialCommunityIcons
                    name={owned ? frame.icon : 'lock'}
                    size={22}
                    color={owned ? frame.ringColor : styles.PALETTE.mutedText}
                  />
                </View>
                <View style={styles.filmModalOptionText}>
                  <Text style={styles.filmModalOptionTitle}>{frame.label}</Text>
                  <Text style={styles.filmModalOptionSub}>{frame.description}</Text>
                </View>
                {selected && <Feather name="check-circle" size={18} color={styles.PALETTE.primary} />}
              </TouchableOpacity>
            );
          })}

          <TouchableOpacity style={styles.filmModalClose} onPress={onClose}>
            <Text style={styles.filmModalCloseText}>{t('common.close')}</Text>
          </TouchableOpacity>
        </View>
      </View>
    </Modal>
  );
}
