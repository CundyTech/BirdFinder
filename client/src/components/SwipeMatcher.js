import React, { useRef, useState } from 'react';
import { View, Text, Image, Animated, PanResponder, Dimensions, TouchableOpacity } from 'react-native';
import { MaterialCommunityIcons, Feather } from '@expo/vector-icons';
import styles from '../styles';
import { useGetSpeciesInfoQuery } from '../services/birdInfoApi';

const SCREEN_WIDTH = Dimensions.get('window').width;
const SWIPE_THRESHOLD = SCREEN_WIDTH * 0.25;
const MAX_ROTATION_DEG = 10;
const FLING_DURATION_MS = 220;

function formatBirdName(className) {
  if (!className) return '';
  return className.replace(/^\d+\./, '').replace(/_/g, ' ');
}

// Reference photo for one candidate — its own small query so each card
// fetches independently (RTK Query dedupes/caches per speciesId same as
// everywhere else in the app).
function CandidateReferenceImage({ candidateId }) {
  const { data: info, isLoading } = useGetSpeciesInfoQuery(candidateId, { skip: !candidateId });
  if (isLoading || !info?.photoUrl) {
    return (
      <View style={[styles.swipeCardImage, styles.swipeCardImagePlaceholder]}>
        <MaterialCommunityIcons name="bird" size={26} color={styles.PALETTE.mutedText} />
      </View>
    );
  }
  return <Image source={{ uri: info.photoUrl }} style={styles.swipeCardImage} />;
}

// Tinder-style card mechanics (drag → rotate → fling-or-snap-back) via
// PanResponder + Animated — deliberately not react-native-gesture-handler
// / reanimated, since this is the only place in the app that needs
// gesture tracking and plain core RN covers it without new dependencies.
// Unlike real Tinder, swipe direction here just cycles which candidate is
// shown — confirming a match is a separate explicit button tap.
export default function SwipeMatcher({ uri, candidates, onMatch }) {
  const [index, setIndex] = useState(0);
  const pan = useRef(new Animated.ValueXY()).current;

  const rotate = pan.x.interpolate({
    inputRange: [-SCREEN_WIDTH / 2, 0, SCREEN_WIDTH / 2],
    outputRange: [`-${MAX_ROTATION_DEG}deg`, '0deg', `${MAX_ROTATION_DEG}deg`],
    extrapolate: 'clamp',
  });

  const advance = (direction) => {
    Animated.timing(pan, {
      toValue: { x: direction * SCREEN_WIDTH * 1.5, y: 0 },
      duration: FLING_DURATION_MS,
      useNativeDriver: false,
    }).start(() => {
      pan.setValue({ x: 0, y: 0 });
      setIndex((current) => (current + direction + candidates.length) % candidates.length);
    });
  };

  const snapBack = () => {
    Animated.spring(pan, {
      toValue: { x: 0, y: 0 },
      friction: 6,
      useNativeDriver: false,
    }).start();
  };

  const panResponder = useRef(
    PanResponder.create({
      onMoveShouldSetPanResponder: (_, gesture) => Math.abs(gesture.dx) > 8,
      onPanResponderMove: Animated.event([null, { dx: pan.x, dy: pan.y }], { useNativeDriver: false }),
      onPanResponderRelease: (_, gesture) => {
        if (gesture.dx < -SWIPE_THRESHOLD) {
          advance(1); // swiped left -> next candidate
        } else if (gesture.dx > SWIPE_THRESHOLD) {
          advance(-1); // swiped right -> previous candidate
        } else {
          snapBack();
        }
      },
    })
  ).current;

  const candidate = candidates[index];
  const candidateName = formatBirdName(candidate.class);

  return (
    <View style={styles.swipeMatcherContainer}>
      <Text style={styles.swipeMatcherTitle}>Which one looks right?</Text>
      <Text style={styles.swipeMatcherHint}>Swipe to compare, then tap "It's a match!"</Text>

      <Animated.View
        {...panResponder.panHandlers}
        style={[
          styles.swipeCard,
          { transform: [{ translateX: pan.x }, { translateY: pan.y }, { rotate }] },
        ]}
      >
        <View style={styles.swipeCardImageRow}>
          <View style={styles.swipeCardImageColumn}>
            <Image source={{ uri }} style={styles.swipeCardImage} />
            <Text style={styles.swipeCardImageCaption}>Your photo</Text>
          </View>
          <View style={styles.swipeCardImageColumn}>
            <CandidateReferenceImage candidateId={candidate.class} />
            <Text style={styles.swipeCardImageCaption}>Reference</Text>
          </View>
        </View>
        <Text style={styles.swipeCardName}>{candidateName}</Text>
      </Animated.View>

      <View style={styles.swipeControlsRow}>
        <TouchableOpacity
          style={styles.swipeArrowButton}
          onPress={() => advance(-1)}
          accessibilityRole="button"
          accessibilityLabel="Previous candidate"
        >
          <Feather name="chevron-left" size={22} color={styles.PALETTE.mutedText} />
        </TouchableOpacity>

        <TouchableOpacity
          style={styles.matchButton}
          onPress={() => onMatch(candidate.class)}
          accessibilityRole="button"
          accessibilityLabel={`Match with ${candidateName}`}
        >
          <MaterialCommunityIcons name="check-bold" size={18} color="#ffffff" />
          <Text style={styles.matchButtonText}>It's a match!</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={styles.swipeArrowButton}
          onPress={() => advance(1)}
          accessibilityRole="button"
          accessibilityLabel="Next candidate"
        >
          <Feather name="chevron-right" size={22} color={styles.PALETTE.mutedText} />
        </TouchableOpacity>
      </View>

      <View style={styles.swipeDotsRow}>
        {candidates.map((c, i) => (
          <View key={c.class || i} style={[styles.swipeDot, i === index && styles.swipeDotActive]} />
        ))}
      </View>
    </View>
  );
}
