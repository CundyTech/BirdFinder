import React, { useState } from 'react';
import { SafeAreaView, ScrollView, View, Text, TouchableOpacity, ActivityIndicator } from 'react-native';
import { Feather } from '@expo/vector-icons';
import styles from '../styles';
import useTrophies from '../hooks/useTrophies';
import TrophyCard from '../components/TrophyCard';

export default function TrophyCabinetScreen({ onBack, onOpenSpecies }) {
  const trophies = useTrophies();
  const [expandedLabel, setExpandedLabel] = useState(null);

  const unlockedCount = trophies ? trophies.filter((t) => t.unlocked).length : 0;

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
          <Text style={styles.subHeaderTitle}>Trophy Cabinet</Text>
          <Text style={styles.subHeaderSubtitle}>
            {trophies ? `${unlockedCount} / ${trophies.length} trophies earned` : 'Loading...'}
          </Text>
        </View>
      </View>

      <ScrollView
        contentContainerStyle={styles.scrollContainer}
        showsVerticalScrollIndicator={false}
        style={styles.mainContent}
      >
        {!trophies ? (
          <View style={styles.referenceLoading}>
            <ActivityIndicator size="small" color={styles.PALETTE.primary} />
            <Text style={styles.referenceLoadingText}>Loading rarity data...</Text>
          </View>
        ) : (
          trophies.map((trophy) => (
            <TrophyCard
              key={trophy.label}
              trophy={trophy}
              expanded={expandedLabel === trophy.label}
              onToggleExpand={() => setExpandedLabel((current) => (current === trophy.label ? null : trophy.label))}
              onOpenSpecies={onOpenSpecies}
            />
          ))
        )}
      </ScrollView>
    </SafeAreaView>
  );
}
