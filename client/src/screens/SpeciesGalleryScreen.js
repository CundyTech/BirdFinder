import React, { useMemo, useState } from 'react';
import { SafeAreaView, ScrollView, View, Text, TouchableOpacity, Image, Alert } from 'react-native';
import { useDispatch, useSelector } from 'react-redux';
import { Feather } from '@expo/vector-icons';
import { useTranslation } from 'react-i18next';
import styles from '../styles';
import { formatSpeciesName } from '../domain/species';
import ImageLightbox from '../components/ImageLightbox';
import SpeciesFactsCard from '../components/SpeciesFactsCard';
import { deleteSighting } from '../store/lifeListSlice';

export default function SpeciesGalleryScreen({ speciesId, onBack }) {
  const { t } = useTranslation();
  const [lightboxUri, setLightboxUri] = useState(null);
  const speciesName = formatSpeciesName(speciesId);
  const dispatch = useDispatch();

  const sightings = useSelector((state) => state.lifeList.sightings);
  const catches = useMemo(
    () => sightings.filter((s) => s.speciesId === speciesId),
    [sightings, speciesId]
  );

  const confirmDelete = (sighting) => {
    Alert.alert(
      t('speciesGallery.deletePhotoTitle'),
      t('speciesGallery.deletePhotoMessage', { speciesName }),
      [
        { text: t('common.cancel'), style: 'cancel' },
        {
          text: t('speciesGallery.delete'),
          style: 'destructive',
          onPress: () => {
            if (lightboxUri === sighting.photoUri) setLightboxUri(null);
            dispatch(deleteSighting(sighting.id));
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
          accessibilityLabel={t('speciesGallery.backToSightingsLog')}
        >
          <Feather name="chevron-left" size={22} color={styles.PALETTE.textOnDark} />
        </TouchableOpacity>
        <View style={styles.subHeaderTitleWrap}>
          <Text style={styles.subHeaderTitle}>{speciesName}</Text>
          <Text style={styles.subHeaderSubtitle}>
            {t('speciesGallery.photoCount', { count: catches.length })}
          </Text>
        </View>
      </View>

      <ScrollView
        contentContainerStyle={styles.scrollContainer}
        showsVerticalScrollIndicator={false}
        style={styles.mainContent}
      >
        <SpeciesFactsCard speciesId={speciesId} speciesName={speciesName} />

        <Text style={styles.sectionLabel}>{t('speciesGallery.sectionLabel')}</Text>

        {catches.length === 0 ? (
          <Text style={styles.emptyStateText}>{t('speciesGallery.emptyStateText')}</Text>
        ) : (
          <View style={styles.galleryGrid}>
            {catches.map((sighting) => (
              <View key={sighting.id} style={styles.galleryGridTile}>
                <TouchableOpacity
                  style={styles.galleryGridTileTouchable}
                  onPress={() => setLightboxUri(sighting.photoUri)}
                  accessibilityRole="imagebutton"
                  accessibilityLabel={t('speciesGallery.viewPhotoOf', { speciesName })}
                >
                  <Image source={{ uri: sighting.photoUri }} style={styles.galleryGridImage} />
                </TouchableOpacity>
                <TouchableOpacity
                  style={styles.galleryDeleteButton}
                  onPress={() => confirmDelete(sighting)}
                  hitSlop={{ top: 8, bottom: 8, left: 8, right: 8 }}
                  accessibilityRole="button"
                  accessibilityLabel={t('speciesGallery.deletePhotoOf', { speciesName })}
                >
                  <Feather name="trash-2" size={14} color="#ffffff" />
                </TouchableOpacity>
              </View>
            ))}
          </View>
        )}
      </ScrollView>

      <ImageLightbox
        uri={lightboxUri}
        label={t('speciesGallery.photoOfFullScreen', { speciesName })}
        onClose={() => setLightboxUri(null)}
      />
    </SafeAreaView>
  );
}
