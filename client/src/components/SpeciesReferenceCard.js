import React, { useState } from 'react';
import { View, Text, Image, ActivityIndicator, TouchableOpacity, Linking } from 'react-native';
import styles from '../styles';
import { useGetSpeciesInfoQuery } from '../services/birdInfoApi';
import ImageLightbox from './ImageLightbox';

export default function SpeciesReferenceCard({ commonName, yourPhotoUri }) {
  const [viewer, setViewer] = useState(null); // { uri, label } | null
  const { data: info, isLoading, isError, refetch } = useGetSpeciesInfoQuery(commonName, {
    skip: !commonName,
  });

  if (!commonName) return null;

  if (isLoading) {
    return (
      <View style={styles.referenceLoading}>
        <ActivityIndicator size="small" color={styles.PALETTE.primary} />
        <Text style={styles.referenceLoadingText}>Looking up species info...</Text>
      </View>
    );
  }

  if (isError) {
    return (
      <View style={styles.referenceErrorCard}>
        <Text style={styles.referenceErrorText}>Couldn't load extra species info.</Text>
        <TouchableOpacity onPress={() => refetch()} accessibilityRole="button" accessibilityLabel="Retry loading species info">
          <Text style={styles.referenceErrorRetry}>Retry</Text>
        </TouchableOpacity>
      </View>
    );
  }

  const hasContent = Boolean(info && (info.photoUrl || info.summary || info.scientificName));
  if (!hasContent) return null;

  const displayName = info.commonName || commonName;

  const stats = [
    info.scientificName && { key: 'scientific', label: 'Scientific name', value: info.scientificName },
    info.family && { key: 'family', label: 'Family', value: info.family },
    info.order && { key: 'order', label: 'Order', value: info.order },
    info.conservationStatus && { key: 'status', label: 'Conservation status', value: info.conservationStatus, isStatus: true },
    typeof info.observationsCount === 'number' && info.observationsCount > 0 && {
      key: 'observations',
      label: 'iNaturalist sightings',
      value: info.observationsCount.toLocaleString(),
    },
  ].filter(Boolean);

  return (
    <View style={styles.referenceCard}>
      <View style={styles.subImageRow}>
        <View style={styles.subImageColumn}>
          <TouchableOpacity
            style={styles.subImageWrapper}
            onPress={() => setViewer({ uri: yourPhotoUri, label: 'Your photo, full screen' })}
            accessibilityRole="imagebutton"
            accessibilityLabel="View your photo larger"
          >
            <Image source={{ uri: yourPhotoUri }} style={styles.subImage} />
          </TouchableOpacity>
          <Text style={styles.subImageLabel}>Your Photo</Text>
        </View>
        {info.photoUrl && (
          <View style={styles.subImageColumn}>
            <TouchableOpacity
              style={styles.subImageWrapper}
              onPress={() => setViewer({ uri: info.photoUrl, label: `Reference photo of ${displayName}, full screen` })}
              accessibilityRole="imagebutton"
              accessibilityLabel={`View larger reference photo of ${displayName}`}
            >
              <Image source={{ uri: info.photoUrl }} style={styles.subImage} />
            </TouchableOpacity>
            <Text style={styles.subImageLabel}>Reference</Text>
          </View>
        )}
      </View>

      {info.photoAttribution && (
        <Text style={styles.referenceAttribution}>Photo: {info.photoAttribution}</Text>
      )}

      {info.galleryPhotos && info.galleryPhotos.length > 0 && (
        <View style={styles.galleryRow}>
          {info.galleryPhotos.map((url, i) => (
            <TouchableOpacity
              key={url}
              style={styles.galleryThumb}
              onPress={() => setViewer({ uri: url, label: `Additional photo of ${displayName}, full screen` })}
              accessibilityRole="imagebutton"
              accessibilityLabel={`View additional photo ${i + 1} of ${displayName} larger`}
            >
              <Image source={{ uri: url }} style={styles.galleryThumbImage} />
            </TouchableOpacity>
          ))}
        </View>
      )}

      {stats.length > 0 && (
        <View style={styles.referenceStatsGrid}>
          {stats.map((stat, i) => (
            <View
              key={stat.key}
              style={[styles.referenceStatItem, i === stats.length - 1 && styles.referenceStatItemLast]}
            >
              <Text style={styles.referenceStatLabel}>{stat.label}</Text>
              <Text style={[styles.referenceStatValue, stat.isStatus && styles.referenceStatStatus]}>
                {stat.value}
              </Text>
            </View>
          ))}
        </View>
      )}

      {info.range && info.range.length > 0 && (
        <View style={styles.rangeSection}>
          {info.range.map((group) => (
            <View key={group.meansLabel} style={styles.rangeRow}>
              <Text style={styles.rangeLabel}>{group.meansLabel} to</Text>
              <Text style={styles.rangeValue}>{group.placesText}</Text>
            </View>
          ))}
        </View>
      )}

      {info.summary ? <Text style={styles.referenceSummary}>{info.summary}</Text> : null}

      {info.wikipediaUrl && (
        <TouchableOpacity onPress={() => Linking.openURL(info.wikipediaUrl)} accessibilityRole="link">
          <Text style={styles.wikiLink}>Read more on Wikipedia →</Text>
        </TouchableOpacity>
      )}

      <ImageLightbox uri={viewer?.uri || null} label={viewer?.label} onClose={() => setViewer(null)} />
    </View>
  );
}
