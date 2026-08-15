import React from 'react';
import { View, Text, TouchableOpacity, Image } from 'react-native';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import styles from '../styles';
import useSpeciesRarity from '../hooks/useSpeciesRarity';
import RarityMeter from './RarityMeter';

// All 60 tiles share the same no-arg rarity query — RTK Query dedupes it
// into one bulk request/cache entry rather than each tile firing its own.
export default function SpeciesTile({ species, count, thumbnailUri, onPress }) {
  const rarity = useSpeciesRarity(species.id);

  return (
    <TouchableOpacity
      style={[styles.speciesTile, count > 0 && styles.speciesTileCaught]}
      onPress={() => count > 0 && onPress(species.id)}
      activeOpacity={count > 0 ? 0.8 : 1}
      disabled={count === 0}
    >
      <View style={styles.speciesTileImageWrapper}>
        {thumbnailUri ? (
          <Image source={{ uri: thumbnailUri }} style={styles.speciesTileImage} />
        ) : (
          <MaterialCommunityIcons name="bird" size={28} color={styles.PALETTE.mutedText} />
        )}
        {count > 0 && (
          <View style={styles.speciesTileCountBadge}>
            <Text style={styles.speciesTileCountBadgeText}>×{count}</Text>
          </View>
        )}
      </View>
      <Text
        style={[styles.speciesTileName, count === 0 && styles.speciesTileNameUncaught]}
        numberOfLines={1}
      >
        {species.name}
      </Text>
      <RarityMeter rarity={rarity} compact />
    </TouchableOpacity>
  );
}
