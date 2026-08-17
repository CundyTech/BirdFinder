import React from 'react';
import { Modal, View, Text, TouchableOpacity } from 'react-native';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import styles from '../styles';
import { useFilmRewardedAd } from '../hooks/useFilmRewardedAd';
import { usePurchases } from '../hooks/usePurchases';

export default function OutOfFilmModal({ visible, onClose }) {
  const { isLoaded: adLoaded, showAd } = useFilmRewardedAd();
  const { product, purchasing, purchaseUnlock } = usePurchases();

  return (
    <Modal visible={visible} transparent animationType="fade" onRequestClose={onClose}>
      <View style={styles.filmModalBackdrop}>
        <View style={styles.filmModalCard}>
          <Text style={styles.filmModalTitle}>Out of Film</Text>
          <Text style={styles.filmModalSubtitle}>
            You're out of Film for now. Grab more below, or come back tomorrow for a free refill.
          </Text>

          <TouchableOpacity
            style={[styles.filmModalOption, !adLoaded && styles.filmModalOptionDisabled]}
            onPress={showAd}
            disabled={!adLoaded}
            activeOpacity={0.85}
          >
            <View style={[styles.filmModalOptionIcon, { backgroundColor: 'rgba(31, 157, 107, 0.16)' }]}>
              <MaterialCommunityIcons name="play-circle-outline" size={22} color={styles.PALETTE.primary} />
            </View>
            <View style={styles.filmModalOptionText}>
              <Text style={styles.filmModalOptionTitle}>Watch an ad</Text>
              <Text style={styles.filmModalOptionSub}>
                {adLoaded ? '+5 Film, free' : 'Loading ad...'}
              </Text>
            </View>
          </TouchableOpacity>

          <TouchableOpacity
            style={[styles.filmModalOption, purchasing && styles.filmModalOptionDisabled]}
            onPress={purchaseUnlock}
            disabled={purchasing}
            activeOpacity={0.85}
          >
            <View style={[styles.filmModalOptionIcon, { backgroundColor: 'rgba(245, 158, 11, 0.16)' }]}>
              <MaterialCommunityIcons name="infinity" size={22} color={styles.PALETTE.accent} />
            </View>
            <View style={styles.filmModalOptionText}>
              <Text style={styles.filmModalOptionTitle}>Unlock unlimited Film</Text>
              <Text style={styles.filmModalOptionSub}>
                {purchasing ? 'Processing...' : `One-time purchase, removes ads too${product?.localizedPrice ? ` — ${product.localizedPrice}` : ''}`}
              </Text>
            </View>
          </TouchableOpacity>

          <TouchableOpacity style={styles.filmModalClose} onPress={onClose}>
            <Text style={styles.filmModalCloseText}>Maybe later</Text>
          </TouchableOpacity>
        </View>
      </View>
    </Modal>
  );
}
