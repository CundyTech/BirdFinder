import React from 'react';
import { View, Text, Image, ActivityIndicator, TouchableOpacity } from 'react-native';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import styles from '../styles';
import { useGetSpeciesInfoQuery } from '../services/birdInfoApi';
import useSpeciesRarity from '../hooks/useSpeciesRarity';
import RarityMeter from './RarityMeter';

export default function SpeciesFactsCard({ speciesId, speciesName }) {
  const { data: info, isLoading, isError, refetch } = useGetSpeciesInfoQuery(speciesName, {
    skip: !speciesName,
  });
  const rarity = useSpeciesRarity(speciesId);

  if (!speciesName) return null;

  if (isLoading) {
    return (
      <View style={styles.referenceLoading}>
        <ActivityIndicator size="small" color={styles.PALETTE.primary} />
        <Text style={styles.referenceLoadingText}>Looking up species facts...</Text>
      </View>
    );
  }

  if (isError) {
    return (
      <View style={styles.referenceErrorCard}>
        <Text style={styles.referenceErrorText}>Couldn't load species facts.</Text>
        <TouchableOpacity onPress={() => refetch()} accessibilityRole="button" accessibilityLabel="Retry loading species facts">
          <Text style={styles.referenceErrorRetry}>Retry</Text>
        </TouchableOpacity>
      </View>
    );
  }

  if (!info) return null;

  const stats = [
    info.family && { key: 'family', label: 'Family', value: info.family },
    info.order && { key: 'order', label: 'Order', value: info.order },
    info.conservationStatus && { key: 'status', label: 'Conservation status', value: info.conservationStatus, isStatus: true },
  ].filter(Boolean);

  return (
    <View style={styles.factsCard}>
      <View style={styles.factsCardTopRow}>
        {info.photoUrl ? (
          <Image source={{ uri: info.photoUrl }} style={styles.factsCardImage} />
        ) : (
          <View style={[styles.factsCardImage, styles.factsCardImagePlaceholder]}>
            <MaterialCommunityIcons name="bird" size={28} color={styles.PALETTE.mutedText} />
          </View>
        )}
        <View style={styles.factsCardHeaderText}>
          <Text style={styles.factsCardName}>{info.commonName || speciesName}</Text>
          {info.scientificName && <Text style={styles.factsCardSciName}>{info.scientificName}</Text>}
        </View>
      </View>

      <RarityMeter rarity={rarity} />

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
    </View>
  );
}
