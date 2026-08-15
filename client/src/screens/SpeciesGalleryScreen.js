import React, { useMemo, useState } from 'react';
import { SafeAreaView, ScrollView, View, Text, TouchableOpacity, Image, Alert } from 'react-native';
import { useDispatch, useSelector } from 'react-redux';
import { Feather } from '@expo/vector-icons';
import styles from '../styles';
import { formatSpeciesName } from '../species';
import ImageLightbox from '../components/ImageLightbox';
import SpeciesFactsCard from '../components/SpeciesFactsCard';
import { deleteSighting } from '../store/lifeListSlice';

export default function SpeciesGalleryScreen({ speciesId, onBack }) {
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
      'Delete photo?',
      `This removes your ${speciesName} photo from the sightings log. This can't be undone.`,
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Delete',
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
          accessibilityLabel="Back to sightings log"
        >
          <Feather name="chevron-left" size={22} color={styles.PALETTE.textOnDark} />
        </TouchableOpacity>
        <View style={styles.subHeaderTitleWrap}>
          <Text style={styles.subHeaderTitle}>{speciesName}</Text>
          <Text style={styles.subHeaderSubtitle}>
            {catches.length} {catches.length === 1 ? 'photo' : 'photos'}
          </Text>
        </View>
      </View>

      <ScrollView
        contentContainerStyle={styles.scrollContainer}
        showsVerticalScrollIndicator={false}
        style={styles.mainContent}
      >
        <SpeciesFactsCard speciesId={speciesId} speciesName={speciesName} />

        <Text style={styles.sectionLabel}>Your snaps</Text>

        {catches.length === 0 ? (
          <Text style={styles.emptyStateText}>No photos of this species yet.</Text>
        ) : (
          <View style={styles.galleryGrid}>
            {catches.map((sighting) => (
              <View key={sighting.id} style={styles.galleryGridTile}>
                <TouchableOpacity
                  style={styles.galleryGridTileTouchable}
                  onPress={() => setLightboxUri(sighting.photoUri)}
                  accessibilityRole="imagebutton"
                  accessibilityLabel={`View photo of ${speciesName}`}
                >
                  <Image source={{ uri: sighting.photoUri }} style={styles.galleryGridImage} />
                </TouchableOpacity>
                <TouchableOpacity
                  style={styles.galleryDeleteButton}
                  onPress={() => confirmDelete(sighting)}
                  hitSlop={{ top: 8, bottom: 8, left: 8, right: 8 }}
                  accessibilityRole="button"
                  accessibilityLabel={`Delete this photo of ${speciesName}`}
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
        label={`Your photo of ${speciesName}, full screen`}
        onClose={() => setLightboxUri(null)}
      />
    </SafeAreaView>
  );
}
