import React, { useMemo } from 'react';
import { SafeAreaView, ScrollView, View, Text, TouchableOpacity } from 'react-native';
import { useSelector } from 'react-redux';
import { Feather } from '@expo/vector-icons';
import styles from '../styles';
import { SPECIES, SPECIES_COUNT } from '../species';
import SpeciesTile from '../components/SpeciesTile';

export default function LifeListScreen({ onBack, onOpenSpecies }) {
  const sightings = useSelector((state) => state.lifeList.sightings);

  const bySpecies = useMemo(() => {
    const map = new Map();
    for (const sighting of sightings) {
      const list = map.get(sighting.speciesId) || [];
      list.push(sighting);
      map.set(sighting.speciesId, list);
    }
    return map;
  }, [sightings]);

  const caughtCount = useMemo(
    () => SPECIES.filter((s) => bySpecies.has(s.id)).length,
    [bySpecies]
  );
  const progress = SPECIES_COUNT > 0 ? caughtCount / SPECIES_COUNT : 0;

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.subHeader}>
        <TouchableOpacity
          style={styles.subHeaderBackButton}
          onPress={onBack}
          accessibilityRole="button"
          accessibilityLabel="Back"
        >
          <Feather name="chevron-left" size={22} color={styles.PALETTE.textOnDark} />
        </TouchableOpacity>
        <View style={styles.subHeaderTitleWrap}>
          <Text style={styles.subHeaderTitle}>My Sightings Log</Text>
          <Text style={styles.subHeaderSubtitle}>{caughtCount} / {SPECIES_COUNT} species spotted</Text>
        </View>
      </View>

      <View style={styles.progressBarTrack}>
        <View style={[styles.progressBarFill, { width: `${Math.round(progress * 100)}%` }]} />
      </View>

      <ScrollView
        contentContainerStyle={styles.scrollContainer}
        showsVerticalScrollIndicator={false}
        style={styles.mainContent}
      >
        <View style={styles.speciesGrid}>
          {SPECIES.map((species) => {
            const catches = bySpecies.get(species.id);
            const count = catches ? catches.length : 0;
            const thumbnailUri = catches ? catches[0].photoUri : null;

            return (
              <SpeciesTile
                key={species.id}
                species={species}
                count={count}
                thumbnailUri={thumbnailUri}
                onPress={onOpenSpecies}
              />
            );
          })}
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}
