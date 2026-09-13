import React from 'react';
import { Modal, View, Text, TouchableOpacity } from 'react-native';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import { useDispatch } from 'react-redux';
import { useTranslation } from 'react-i18next';
import styles from '../styles';
import { useFilmRewardedAd } from '../hooks/useFilmRewardedAd';
import { usePurchases } from '../hooks/usePurchases';
import { grantAdReward } from '../store/filmSlice';

export default function OutOfFilmModal({ visible, onClose, rerollTokens = 0, onUseReroll }) {
  const { t } = useTranslation();
  const dispatch = useDispatch();
  const { isLoaded: adLoaded, showAd } = useFilmRewardedAd(() => dispatch(grantAdReward()));
  const { product, purchasing, purchaseUnlock } = usePurchases();

  return (
    <Modal visible={visible} transparent animationType="fade" onRequestClose={onClose}>
      <View style={styles.filmModalBackdrop}>
        <View style={styles.filmModalCard}>
          <Text style={styles.filmModalTitle}>{t('modals.outOfFilm.title')}</Text>
          <Text style={styles.filmModalSubtitle}>{t('modals.outOfFilm.subtitle')}</Text>

          {rerollTokens > 0 && (
            <TouchableOpacity style={styles.filmModalOption} onPress={onUseReroll} activeOpacity={0.85}>
              <View style={[styles.filmModalOptionIcon, { backgroundColor: 'rgba(31, 157, 107, 0.16)' }]}>
                <MaterialCommunityIcons name="dice-multiple-outline" size={22} color={styles.PALETTE.primary} />
              </View>
              <View style={styles.filmModalOptionText}>
                <Text style={styles.filmModalOptionTitle}>{t('modals.outOfFilm.useReroll')}</Text>
                <Text style={styles.filmModalOptionSub}>
                  {t('modals.outOfFilm.rerollTokensAvailable', { count: rerollTokens })}
                </Text>
              </View>
            </TouchableOpacity>
          )}

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
              <Text style={styles.filmModalOptionTitle}>{t('modals.outOfFilm.watchAd')}</Text>
              <Text style={styles.filmModalOptionSub}>
                {adLoaded ? t('modals.outOfFilm.adFreeFilm') : t('common.loadingAd')}
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
              <Text style={styles.filmModalOptionTitle}>{t('modals.outOfFilm.unlockUnlimited')}</Text>
              <Text style={styles.filmModalOptionSub}>
                {purchasing
                  ? t('modals.outOfFilm.processing')
                  : t('modals.outOfFilm.unlockSub', { price: product?.localizedPrice ? ` — ${product.localizedPrice}` : '' })}
              </Text>
            </View>
          </TouchableOpacity>

          <TouchableOpacity style={styles.filmModalClose} onPress={onClose}>
            <Text style={styles.filmModalCloseText}>{t('modals.outOfFilm.maybeLater')}</Text>
          </TouchableOpacity>
        </View>
      </View>
    </Modal>
  );
}
