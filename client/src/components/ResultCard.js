import React, { useState } from 'react';
import { View, Text, Image, TouchableOpacity } from 'react-native';
import { Feather } from '@expo/vector-icons';
import styles from '../styles';
import SpeciesFactsCard from './SpeciesFactsCard';
import SwipeMatcher from './SwipeMatcher';
import ImageLightbox from './ImageLightbox';
import { LOW_CONFIDENCE_THRESHOLD } from '../config';

export default function ResultCard({ uri, result, onSave }) {
  const [viewerOpen, setViewerOpen] = useState(false);
  // null until the user taps/matches a different candidate — falls back
  // to the top prediction so "Save" has a sensible default without
  // assuming it.
  const [selectedClass, setSelectedClass] = useState(null);
  const [saved, setSaved] = useState(false);
  // Only ever set via the swipe matcher — lets the profile view below
  // distinguish "confident from the start" from "resolved via swiping",
  // independent of whatever confidence the matched candidate itself has.
  const [matched, setMatched] = useState(false);

  if (!result) return null;

  const getMaxScore = () => {
    if (!result || !result.scores || !result.scores.length) return 0;
    return Math.max(...result.scores);
  };

  const formatBirdName = (className) => {
    if (!className) return '';
    const name = className.replace(/^\d+\./, '').replace(/_/g, ' ');
    return name;
  };

  const topPredictions = Array.isArray(result.top_predictions) && result.top_predictions.length > 0
    ? result.top_predictions
    : [{ class: result.predicted_class, score: getMaxScore() }];
  const hasAlternatives = topPredictions.length > 1;

  const lightbox = (
    <ImageLightbox uri={viewerOpen ? uri : null} label="Your photo, full screen" onClose={() => setViewerOpen(false)} />
  );

  // Everything below — title, confidence badge, reference card — reflects
  // whichever candidate is selected, not just the model's top-1 guess, so
  // switching candidates updates the whole page, not just the save button.
  const effectiveSelectedClass = selectedClass || result.predicted_class;
  const selectedCandidate =
    topPredictions.find((p) => p.class === effectiveSelectedClass) || {
      class: result.predicted_class,
      score: getMaxScore(),
    };
  const selectedName = formatBirdName(effectiveSelectedClass);
  const displayScorePercent = Number(((selectedCandidate.score || 0) * 100).toFixed(1));
  const isLowConfidence = displayScorePercent < LOW_CONFIDENCE_THRESHOLD;
  const confidenceLabel = isLowConfidence ? 'Possible match' : 'Strong match';

  const handleSave = () => {
    if (saved) return;
    onSave?.(effectiveSelectedClass, selectedCandidate.score || 0);
    setSaved(true);
  };

  // Anything below LOW_CONFIDENCE_THRESHOLD goes through the swipe matcher
  // as long as there's more than one candidate to compare — swiping
  // through a single photo has nothing to switch between, so that case
  // (and any confident result) goes straight to the profile view instead.
  if (isLowConfidence && hasAlternatives && !matched) {
    return (
      <View style={styles.resultCard}>
        <SwipeMatcher
          uri={uri}
          candidates={topPredictions.slice(0, 3)}
          onMatch={(candidateClass) => {
            setSelectedClass(candidateClass);
            setMatched(true);
          }}
        />
        {lightbox}
      </View>
    );
  }

  return (
    <View style={styles.resultCard}>
      {!matched && (
        <View style={styles.confidenceBadgeRow}>
          <View style={[styles.confidenceBadge, isLowConfidence && styles.confidenceBadgeLow]}>
            <Text style={[styles.confidenceText, isLowConfidence && styles.confidenceTextLow]}>{confidenceLabel}</Text>
          </View>
        </View>
      )}

      {uri && (
        <TouchableOpacity
          style={styles.resultHeroImageWrapper}
          onPress={() => setViewerOpen(true)}
          accessibilityRole="imagebutton"
          accessibilityLabel="View your photo larger"
        >
          <Image source={{ uri }} style={styles.resultHeroImage} />
        </TouchableOpacity>
      )}

      <SpeciesFactsCard speciesId={effectiveSelectedClass} speciesName={selectedName} />

      {matched && hasAlternatives && !saved && (
        <TouchableOpacity
          onPress={() => setMatched(false)}
          accessibilityRole="button"
          accessibilityLabel="Not the right bird, choose again"
        >
          <Text style={styles.swipeAgainLink}>Not right? Choose again</Text>
        </TouchableOpacity>
      )}

      <TouchableOpacity
        style={[styles.saveButton, saved && styles.saveButtonSaved]}
        onPress={handleSave}
        disabled={saved}
        accessibilityRole="button"
        accessibilityLabel={saved ? 'Saved to sightings log' : `Save ${selectedName} to sightings log`}
      >
        <Feather
          name={saved ? 'check-circle' : 'bookmark'}
          size={18}
          color={saved ? styles.PALETTE.primary : '#ffffff'}
        />
        <Text style={[styles.saveButtonText, saved && styles.saveButtonTextSaved]}>
          {saved ? `Saved as ${selectedName}` : `Save as ${selectedName}`}
        </Text>
      </TouchableOpacity>

      {lightbox}
    </View>
  );
}
