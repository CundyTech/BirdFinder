import React from 'react';
import { View, Text, TouchableOpacity, Image } from 'react-native';
import { MaterialCommunityIcons, Feather } from '@expo/vector-icons';
import { useTranslation } from 'react-i18next';
import styles from '../styles';
import { rewardDisplay } from '../domain/rewardDisplay';

function SpeciesTrophyBody({ trophy, onOpenSpecies }) {
  const { species, caughtSpeciesIds } = trophy;
  return (
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
  );
}

function BehaviorTrophyBody({ trophy, onOpenSpecies }) {
  const { t } = useTranslation();
  if (!trophy.qualifyingSightings || trophy.qualifyingSightings.length === 0) {
    return <Text style={styles.trophyCategoryDescription}>{trophy.description}</Text>;
  }
  return (
    <View style={styles.galleryGrid}>
      {trophy.qualifyingSightings.map((sighting) => (
        <TouchableOpacity
          key={sighting.id}
          style={styles.galleryGridTile}
          onPress={() => onOpenSpecies(sighting.speciesId)}
          accessibilityRole="imagebutton"
          accessibilityLabel={t('components.trophyCard.viewSightingSpecies')}
        >
          <View style={styles.galleryGridTileTouchable}>
            <Image source={{ uri: sighting.photoUri }} style={styles.galleryGridImage} />
          </View>
        </TouchableOpacity>
      ))}
    </View>
  );
}

export default function TrophyCard({ trophy, expanded, onToggleExpand, onOpenSpecies }) {
  const { t } = useTranslation();
  const { type, label, unlocked } = trophy;
  const isBehavior = type === 'behavior';

  const current = isBehavior ? trophy.current : trophy.caughtCount;
  const target = isBehavior ? trophy.target : trophy.total;
  const progress = target > 0 ? Math.min(current / target, 1) : 0;
  const progressText = isBehavior
    ? t('components.trophyCard.progressBehavior', { current, target, unitLabel: trophy.unitLabel })
    : t('components.trophyCard.progressSpecies', { current, target });
  const reward = unlocked ? rewardDisplay(trophy.reward) : null;

  return (
    <View style={[styles.trophyCard, unlocked && styles.trophyCardUnlocked]}>
      <TouchableOpacity
        style={styles.trophyCardHeader}
        onPress={onToggleExpand}
        accessibilityRole="button"
        accessibilityLabel={t('components.trophyCard.accessibilityLabel', {
          label,
          progressText,
          action: expanded ? t('components.trophyCard.collapse') : t('components.trophyCard.expand'),
        })}
      >
        <View style={styles.trophyIconWrap}>
          <View style={[styles.trophyIconCircle, unlocked && styles.trophyIconCircleUnlocked]}>
            <MaterialCommunityIcons
              name={unlocked ? 'trophy' : 'trophy-outline'}
              size={24}
              color={unlocked ? styles.PALETTE.accent : styles.PALETTE.mutedText}
            />
          </View>
          {reward && (
            <View style={[styles.trophyRewardBadge, { backgroundColor: reward.color }]}>
              <MaterialCommunityIcons name={reward.icon} size={12} color="#0e1116" />
            </View>
          )}
        </View>
        <View style={styles.trophyHeaderText}>
          <Text style={styles.trophyLabel}>{label}</Text>
          <Text style={styles.trophyProgressText}>{progressText}</Text>
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
        <View style={styles.trophyExpandedBody}>
          {isBehavior ? (
            <BehaviorTrophyBody trophy={trophy} onOpenSpecies={onOpenSpecies} />
          ) : (
            <SpeciesTrophyBody trophy={trophy} onOpenSpecies={onOpenSpecies} />
          )}
        </View>
      )}
    </View>
  );
}
