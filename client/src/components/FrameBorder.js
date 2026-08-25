import React from 'react';
import { View } from 'react-native';
import styles from '../styles';

// Renders the equipped cosmetic frame's hand-drawn SVG art on top of a
// photo (see domain/frames.js). Must sit inside a same-sized parent whose
// aspect ratio already matches frame.viewBox — see ImageLightbox.js, which
// sizes that parent and positions the photo inside the frame's window.
export default function FrameBorder({ frame }) {
  if (!frame) return null;
  const { Svg } = frame;

  return (
    <View pointerEvents="none" style={styles.frameBorderFill}>
      <Svg width="100%" height="100%" />
    </View>
  );
}
