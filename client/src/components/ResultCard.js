import React, { useState } from 'react';
import { View, Text, Image, TouchableOpacity } from 'react-native';
import { Feather } from '@expo/vector-icons';
import styles from '../styles';
import SpeciesReferenceCard from './SpeciesReferenceCard';
import ImageLightbox from './ImageLightbox';
import { LOW_CONFIDENCE_THRESHOLD, UNCERTAIN_THRESHOLD, DEBUG_ALWAYS_SHOW_SPECIES_DETAILS } from '../config';

export default function ResultCard({ uri, result }) {
  const [viewerOpen, setViewerOpen] = useState(false);

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

  const headerCandidate = { class: result.predicted_class, score: getMaxScore() };
  const displayScorePercent = Number((headerCandidate.score * 100).toFixed(1));
  const headerName = formatBirdName(headerCandidate.class);
  const isLowConfidence = displayScorePercent < LOW_CONFIDENCE_THRESHOLD;
  const confidenceLabel = isLowConfidence ? 'Possible match' : 'Strong match';

  const topPredictions = Array.isArray(result.top_predictions) && result.top_predictions.length > 0
    ? result.top_predictions
    : [{ class: result.predicted_class, score: getMaxScore() }];

  return (
    <View style={styles.resultCard}>
      <View style={styles.resultHeaderRow}>
        <Text style={styles.birdName}>{headerName}</Text>
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

      <SpeciesReferenceCard commonName={headerName} yourPhotoUri={uri} />

      <View style={styles.predictionsCard}>
        <Text style={styles.predictionsTitle}>Similar species</Text>
        {topPredictions.slice(0, 3).map((p, i, arr) => {
          const isLast = i === arr.length - 1;
          const scorePercent = Math.min((p.score || 0) * 100, 100);
          return (
            <View key={p.class || i} style={[styles.predictionRow, isLast && styles.predictionRowLast]}>
              <View style={styles.predictionRowTop}>
                <View style={styles.predictionRank}>
                  <Text style={styles.predictionRankText}>{i + 1}</Text>
                </View>
                <Text style={styles.predictionLabel}>{formatBirdName(p.class)}</Text>
              </View>
              <View style={styles.predictionBarTrack}>
                <View style={[styles.predictionBarFill, { width: `${scorePercent}%` }]} />
              </View>
            </View>
          );
        })}
      </View>

      {lightbox}
    </View>
  );
}
