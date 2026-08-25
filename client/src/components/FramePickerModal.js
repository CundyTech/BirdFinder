import React from 'react';
import { Modal, View, Text, TouchableOpacity } from 'react-native';
import { MaterialCommunityIcons, Feather } from '@expo/vector-icons';
import { useDispatch, useSelector } from 'react-redux';
import styles from '../styles';
import { FRAMES } from '../domain/frames';
import { equipFrame } from '../store/rewardsSlice';

export default function FramePickerModal({ visible, onClose }) {
  const dispatch = useDispatch();
  const ownedFrameIds = useSelector((state) => state.rewards.ownedFrameIds);
  const equippedFrameId = useSelector((state) => state.rewards.equippedFrameId);

  const ownedFrames = FRAMES.filter((f) => ownedFrameIds.includes(f.id));

  return (
    <Modal visible={visible} transparent animationType="fade" onRequestClose={onClose}>
      <View style={styles.filmModalBackdrop}>
        <View style={styles.filmModalCard}>
          <Text style={styles.filmModalTitle}>Your Frames</Text>
          <Text style={styles.filmModalSubtitle}>
            Earned from milestones. Pick one to frame your photos when viewing them full-screen.
          </Text>

          {ownedFrames.map((frame) => {
            const selected = frame.id === equippedFrameId;
            return (
              <TouchableOpacity
                key={frame.id}
                style={styles.filmModalOption}
                onPress={() => dispatch(equipFrame(frame.id))}
                activeOpacity={0.85}
              >
                <View style={[styles.filmModalOptionIcon, { backgroundColor: `${frame.ringColor}29` }]}>
                  <MaterialCommunityIcons name={frame.icon} size={22} color={frame.ringColor} />
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
            <Text style={styles.filmModalCloseText}>Close</Text>
          </TouchableOpacity>
        </View>
      </View>
    </Modal>
  );
}
