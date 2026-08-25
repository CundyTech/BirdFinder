import React from 'react';
import { View, Text, TouchableOpacity, Image } from 'react-native';
import { MaterialCommunityIcons, Feather } from '@expo/vector-icons';
import styles from '../styles';
import { getFrame } from '../domain/frames';

// Milestone trophies (see achievements.js) carry a `reward` spec describing
// what they actually grant. This maps that to a small icon/colour shown
// right on the trophy — the reward itself often lives somewhere less
// obvious (a cosmetic frame only shows on the Home header, for instance),
// so this is the one place every reward is visible regardless of type.
function rewardDisplay(reward) {
  if (!reward) return null;
  if (reward.type === 'reroll') {
    return { icon: 'dice-multiple-outline', color: styles.PALETTE.primary };
  }
  if (reward.type === 'streakProtection') {
    return { icon: 'shield-check-outline', color: styles.PALETTE.primary };
  }
  if (reward.type === 'frame') {
    const frame = getFrame(reward.frameId);
    return { icon: frame?.icon || 'star-four-points', color: frame?.ringColor || styles.PALETTE.accent };
  }
  if (reward.type === 'deepDiveLore') {
    return { icon: 'book-open-page-variant', color: styles.PALETTE.primary };
  }
  return null;
}

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
          accessibilityLabel="View this sighting's species"
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
  const { type, label, unlocked } = trophy;
  const isBehavior = type === 'behavior';

  const current = isBehavior ? trophy.current : trophy.caughtCount;
  const target = isBehavior ? trophy.target : trophy.total;
  const progress = target > 0 ? Math.min(current / target, 1) : 0;
  const progressText = isBehavior ? `${current} / ${target} ${trophy.unitLabel}` : `${current} / ${target} caught`;
  const reward = unlocked ? rewardDisplay(trophy.reward) : null;

  return (
    <View style={[styles.trophyCard, unlocked && styles.trophyCardUnlocked]}>
      <TouchableOpacity
        style={styles.trophyCardHeader}
        onPress={onToggleExpand}
        accessibilityRole="button"
        accessibilityLabel={`${label} trophy, ${progressText}, ${expanded ? 'collapse' : 'expand'} details`}
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
