import React, { useState } from 'react';
import { View, Text, Image, TouchableOpacity } from 'react-native';
import { Feather } from '@expo/vector-icons';
import styles from '../styles';
import SpeciesReferenceCard from './SpeciesReferenceCard';
import ImageLightbox from './ImageLightbox';
import { LOW_CONFIDENCE_THRESHOLD, UNCERTAIN_THRESHOLD, DEBUG_ALWAYS_SHOW_SPECIES_DETAILS } from '../config';

export default function ResultCard({ uri, result, onSave }) {
  const [viewerOpen, setViewerOpen] = useState(false);
  // null until the user taps a different candidate — falls back to the
  // top prediction so "Save" has a sensible default without assuming it.
  const [selectedClass, setSelectedClass] = useState(null);
  const [saved, setSaved] = useState(false);

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

  const topConfidencePercent = Number((getMaxScore() * 100).toFixed(1));

  // Below UNCERTAIN_THRESHOLD we don't auto-guess a species at all — the
  // top-1 pick is too shaky to present as a confident identification. We
  // show tips instead of a name/photo/stats we can't stand behind.
  const isUncertain = !DEBUG_ALWAYS_SHOW_SPECIES_DETAILS && topConfidencePercent < UNCERTAIN_THRESHOLD;

  const lightbox = (
    <ImageLightbox uri={viewerOpen ? uri : null} label="Your photo, full screen" onClose={() => setViewerOpen(false)} />
  );

  if (isUncertain) {
    return (
      <View style={[styles.resultCard, styles.resultCardFill]}>
        <View style={styles.uncertainHeaderRow}>
          <Feather name="camera" size={22} color={styles.PALETTE.accent} />
          <Text style={styles.uncertainHeaderTitle}>We need a clearer look</Text>
        </View>
        <Text style={styles.uncertainHeaderSubtitle}>
          Try getting closer, using even lighting, and holding the camera steady.
        </Text>

        {uri && (
          <TouchableOpacity
            style={styles.uncertainImageWrapper}
            onPress={() => setViewerOpen(true)}
            accessibilityRole="imagebutton"
            accessibilityLabel="View your photo larger"
          >
            <Image source={{ uri }} style={styles.uncertainImage} />
          </TouchableOpacity>
        )}

        {lightbox}
      </View>
    );
  }

  const topPredictions = Array.isArray(result.top_predictions) && result.top_predictions.length > 0
    ? result.top_predictions
    : [{ class: result.predicted_class, score: getMaxScore() }];

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

  return (
    <View style={styles.resultCard}>
      <View style={styles.resultHeaderRow}>
        <Text style={styles.birdName}>{selectedName}</Text>
        <View style={[styles.confidenceBadge, isLowConfidence && styles.confidenceBadgeLow]}>
          <Text style={[styles.confidenceText, isLowConfidence && styles.confidenceTextLow]}>{confidenceLabel}</Text>
        </View>
      </View>

      {isLowConfidence && (
        <View style={styles.lowConfidenceBanner}>
          <Text style={styles.lowConfidenceText}>
            This might not be quite right — try a closer, well-lit photo for a better match.
          </Text>
        </View>
      )}

      <SpeciesReferenceCard commonName={selectedName} yourPhotoUri={uri} />

      <View style={styles.predictionsCard}>
        <Text style={styles.predictionsTitle}>Similar species</Text>
        {topPredictions.length > 1 && (
          <Text style={styles.predictionsHint}>
            Not quite right? Tap the correct match before saving.
          </Text>
        )}
        {topPredictions.slice(0, 3).map((p, i, arr) => {
          const isLast = i === arr.length - 1;
          const scorePercent = Math.min((p.score || 0) * 100, 100);
          const isSelected = p.class === effectiveSelectedClass;
          return (
            <TouchableOpacity
              key={p.class || i}
              style={[
                styles.predictionRow,
                isLast && styles.predictionRowLast,
                isSelected && styles.predictionRowSelected,
              ]}
              onPress={() => setSelectedClass(p.class)}
              disabled={saved}
              accessibilityRole="button"
              accessibilityLabel={`Choose ${formatBirdName(p.class)} as the species to save`}
            >
              <View style={styles.predictionRowTop}>
                <View style={[styles.predictionRank, isSelected && styles.predictionRankSelected]}>
                  {isSelected ? (
                    <Feather name="check" size={14} color="#ffffff" />
                  ) : (
                    <Text style={styles.predictionRankText}>{i + 1}</Text>
                  )}
                </View>
                <Text style={styles.predictionLabel}>{formatBirdName(p.class)}</Text>
              </View>
              <View style={styles.predictionBarTrack}>
                <View style={[styles.predictionBarFill, { width: `${scorePercent}%` }]} />
              </View>
            </TouchableOpacity>
          );
        })}
      </View>

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
