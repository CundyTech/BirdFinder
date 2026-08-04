import React, { useState } from 'react';
import { View, Text, TouchableOpacity } from 'react-native';
import { Feather } from '@expo/vector-icons';
import styles from '../styles';
import SpeciesReferenceCard from './SpeciesReferenceCard';
import { LOW_CONFIDENCE_THRESHOLD, UNCERTAIN_THRESHOLD, DEBUG_ALWAYS_SHOW_SPECIES_DETAILS } from '../config';

export default function ResultCard({ uri, result }) {
  const [selectedCandidate, setSelectedCandidate] = useState(null);

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
  // show tips + a tappable candidate list instead, and only reveal species
  // detail once the user picks one themselves.
  const isUncertain = !DEBUG_ALWAYS_SHOW_SPECIES_DETAILS && topConfidencePercent < UNCERTAIN_THRESHOLD;

  // The backend ranks every class server-side and sends the real top few —
  // falls back to a single entry only if an older backend build is running.
  const topPredictions = Array.isArray(result.top_predictions) && result.top_predictions.length > 0
    ? result.top_predictions
    : [{ class: result.predicted_class, score: getMaxScore() }];

  // What we actually show a name/badge/detail card for: the model's top
  // pick when we trust it, or whatever the user tapped when we don't.
  const headerCandidate = !isUncertain
    ? { class: result.predicted_class, score: getMaxScore() }
    : selectedCandidate;

  const displayScorePercent = headerCandidate
    ? Number((headerCandidate.score * 100).toFixed(1))
    : topConfidencePercent;
  const headerName = headerCandidate ? formatBirdName(headerCandidate.class) : 'Uncertain match';
  const isLowConfidence = displayScorePercent < LOW_CONFIDENCE_THRESHOLD;
  const isUserPick = isUncertain && Boolean(selectedCandidate);

  const predictionsBlock = (
    <View style={styles.predictionsCard}>
      <Text style={styles.predictionsTitle}>
        {isUncertain ? 'Possible matches — tap one to see details' : 'Top predictions'}
      </Text>
      {topPredictions.slice(0, 3).map((p, i, arr) => {
        const isLast = i === arr.length - 1;
        const isSelected = isUncertain && selectedCandidate?.class === p.class;
        const rowStyle = [
          styles.predictionRow,
          isLast && styles.predictionRowLast,
          isSelected && styles.predictionRowSelected,
        ];
        const scorePercent = Math.min((p.score || 0) * 100, 100);
        const row = (
          <>
            <View style={styles.predictionRowTop}>
              <View style={styles.predictionRank}>
                <Text style={styles.predictionRankText}>{i + 1}</Text>
              </View>
              <Text style={styles.predictionLabel}>{formatBirdName(p.class)}</Text>
              <Text style={styles.predictionPercent}>{scorePercent.toFixed(1)}%</Text>
            </View>
            <View style={styles.predictionBarTrack}>
              <View style={[styles.predictionBarFill, { width: `${scorePercent}%` }]} />
            </View>
          </>
        );
        if (!isUncertain) {
          return <View key={p.class || i} style={rowStyle}>{row}</View>;
        }
        return (
          <TouchableOpacity
            key={p.class || i}
            style={rowStyle}
            onPress={() => setSelectedCandidate(isSelected ? null : p)}
            accessibilityRole="button"
            accessibilityState={{ selected: isSelected }}
            accessibilityLabel={`${formatBirdName(p.class)}, ${scorePercent.toFixed(1)} percent confidence${isSelected ? ', selected' : ''}`}
          >
            {row}
          </TouchableOpacity>
        );
      })}
    </View>
  );

  return (
    <View style={styles.resultCard}>
      <View style={styles.resultHeaderRow}>
        <Text style={styles.birdName}>{headerName}</Text>
        <View style={[styles.confidenceBadge, isLowConfidence && styles.confidenceBadgeLow]}>
          <Text style={[styles.confidenceText, isLowConfidence && styles.confidenceTextLow]}>{displayScorePercent}%</Text>
        </View>
      </View>

      {isUncertain ? (
        <View style={styles.tipsCard}>
          <View style={styles.tipsHeaderRow}>
            <Feather name="alert-triangle" size={18} color={styles.PALETTE.accent} />
            <Text style={styles.tipsTitle}>Not confident enough to guess</Text>
          </View>
          <Text style={styles.tipsText}>
            The model isn't sure enough to identify this automatically. For a better result, try:
          </Text>
          <Text style={styles.tipsListItem}>• Getting closer to the bird</Text>
          <Text style={styles.tipsListItem}>• Even, natural lighting</Text>
          <Text style={styles.tipsListItem}>• Holding the camera steady</Text>
          <Text style={styles.tipsListItem}>• Framing the bird without obstructions</Text>
        </View>
      ) : (
        isLowConfidence && (
          <View style={styles.lowConfidenceBanner}>
            <Text style={styles.lowConfidenceText}>
              Low confidence match — try a closer, well-lit photo for a more accurate result.
            </Text>
          </View>
        )
      )}

      {isUncertain && predictionsBlock}

      {isUserPick && (
        <View style={styles.userPickBanner}>
          <Feather name="alert-circle" size={16} color={styles.PALETTE.accent} />
          <Text style={styles.userPickText}>
            You picked this — confidence was only {displayScorePercent}%. This is not a confirmed identification.
          </Text>
        </View>
      )}

      {headerCandidate && (
        <SpeciesReferenceCard commonName={formatBirdName(headerCandidate.class)} yourPhotoUri={uri} />
      )}

      {!isUncertain && predictionsBlock}
    </View>
  );
}
