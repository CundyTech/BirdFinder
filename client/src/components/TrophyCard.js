import React from 'react';
import { View, Text, TouchableOpacity } from 'react-native';
import { MaterialCommunityIcons, Feather } from '@expo/vector-icons';
import styles from '../styles';

export default function TrophyCard({ trophy, expanded, onToggleExpand, onOpenSpecies }) {
  const { label, species, caughtSpeciesIds, caughtCount, total, unlocked } = trophy;
  const progress = total > 0 ? caughtCount / total : 0;

  return (
    <View style={[styles.trophyCard, unlocked && styles.trophyCardUnlocked]}>
      <TouchableOpacity
        style={styles.trophyCardHeader}
        onPress={onToggleExpand}
        accessibilityRole="button"
        accessibilityLabel={`${label} trophy, ${caughtCount} of ${total} caught, ${expanded ? 'collapse' : 'expand'} species list`}
      >
        <View style={[styles.trophyIconCircle, unlocked && styles.trophyIconCircleUnlocked]}>
          <MaterialCommunityIcons
            name={unlocked ? 'trophy' : 'trophy-outline'}
            size={24}
            color={unlocked ? styles.PALETTE.accent : styles.PALETTE.mutedText}
          />
        </View>
        <View style={styles.trophyHeaderText}>
          <Text style={styles.trophyLabel}>{label}</Text>
          <Text style={styles.trophyProgressText}>{caughtCount} / {total} caught</Text>
        </View>
        <Feather
          name={expanded ? 'chevron-up' : 'chevron-down'}
          size={20}
          color={styles.PALETTE.mutedText}
        />
      </TouchableOpacity>

      <View style={styles.trophyProgressBarTrack}>
        <View style={[styles.trophyProgressBarFill, { width: `${Math.round(progress * 100)}%` }]} />
      </View>

      {expanded && (
        <View style={styles.trophySpeciesGrid}>
          {species.map((s) => {
            const caught = caughtSpeciesIds.has(s.id);
            return (
              <TouchableOpacity
                key={s.id}
                style={[styles.trophySpeciesChip, caught && styles.trophySpeciesChipCaught]}
                onPress={() => caught && onOpenSpecies(s.id)}
                activeOpacity={caught ? 0.7 : 1}
                disabled={!caught}
              >
                <Feather
                  name={caught ? 'check-circle' : 'lock'}
                  size={12}
                  color={caught ? styles.PALETTE.primary : styles.PALETTE.mutedText}
                />
                <Text style={[styles.trophySpeciesChipText, caught && styles.trophySpeciesChipTextCaught]}>
                  {s.name}
                </Text>
              </TouchableOpacity>
            );
          })}
        </View>
      )}
    </View>
  );
}
