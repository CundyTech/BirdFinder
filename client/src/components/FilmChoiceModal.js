import React from 'react';
import { Modal, View, Text, TouchableOpacity } from 'react-native';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import styles from '../styles';
import { useFilmRewardedAd } from '../hooks/useFilmRewardedAd';

// Shown every time the user starts an identification while they still have
// Film — not just once it runs out (that's OutOfFilmModal) — so watching an
// ad to cover the shot is always on offer, not just a last resort. Picking
// the ad here doesn't touch the Film balance at all (see onAdEarned): it
// covers this one identification directly, unlike OutOfFilmModal's ad
// option, which tops up the balance for future use.
export default function FilmChoiceModal({ visible, onClose, filmBalance, onUseFilm, onAdEarned }) {
  const { isLoaded: adLoaded, showAd } = useFilmRewardedAd(onAdEarned);

  const handleWatchAd = () => {
    onClose();
    showAd();
  };

  return (
    <Modal visible={visible} transparent animationType="fade" onRequestClose={onClose}>
      <View style={styles.filmModalBackdrop}>
        <View style={styles.filmModalCard}>
          <Text style={styles.filmModalTitle}>Ready to identify?</Text>
          <Text style={styles.filmModalSubtitle}>
            Use a Film, or watch a quick ad to cover this one for free.
          </Text>

          <TouchableOpacity style={styles.filmModalOption} onPress={onUseFilm} activeOpacity={0.85}>
            <View style={[styles.filmModalOptionIcon, { backgroundColor: 'rgba(245, 158, 11, 0.16)' }]}>
              <MaterialCommunityIcons name="filmstrip" size={22} color={styles.PALETTE.accent} />
            </View>
            <View style={styles.filmModalOptionText}>
              <Text style={styles.filmModalOptionTitle}>Use 1 Film</Text>
              <Text style={styles.filmModalOptionSub}>{filmBalance} Film available</Text>
            </View>
          </TouchableOpacity>

          <TouchableOpacity
            style={[styles.filmModalOption, !adLoaded && styles.filmModalOptionDisabled]}
            onPress={handleWatchAd}
            disabled={!adLoaded}
            activeOpacity={0.85}
          >
            <View style={[styles.filmModalOptionIcon, { backgroundColor: 'rgba(31, 157, 107, 0.16)' }]}>
              <MaterialCommunityIcons name="play-circle-outline" size={22} color={styles.PALETTE.primary} />
            </View>
            <View style={styles.filmModalOptionText}>
              <Text style={styles.filmModalOptionTitle}>Watch an ad instead</Text>
              <Text style={styles.filmModalOptionSub}>
                {adLoaded ? 'Covers this identification, free' : 'Loading ad...'}
              </Text>
            </View>
          </TouchableOpacity>

          <TouchableOpacity style={styles.filmModalClose} onPress={onClose}>
            <Text style={styles.filmModalCloseText}>Cancel</Text>
          </TouchableOpacity>
        </View>
      </View>
    </Modal>
  );
}
