import React from 'react';
import { StyleSheet } from 'react-native';
import Svg, { Defs, Pattern, Rect, Path, Circle, Ellipse, Line, G } from 'react-native-svg';

const TILE = 200;
const STROKE = 'rgba(230, 238, 248, 0.06)';
const STROKE_WIDTH = 1.3;

export default function BirdPatternBackground() {
  return (
    <Svg style={StyleSheet.absoluteFill} width="100%" height="100%" pointerEvents="none">
      <Defs>
        <Pattern id="birdDoodles" width={TILE} height={TILE} patternUnits="userSpaceOnUse">
          <G stroke={STROKE} strokeWidth={STROKE_WIDTH} fill="none" strokeLinecap="round" strokeLinejoin="round">
            {/* Flying bird */}
            <Path d="M 15,25 Q 30,7 45,23 Q 60,7 75,25" transform="rotate(-6 45 16)" />

            {/* Perched bird on a branch */}
            <G transform="translate(120,15)">
              <Path d="M -5,26 L 35,26" />
              <Ellipse cx="10" cy="14" rx="11" ry="9" />
              <Circle cx="20" cy="6" r="5" />
              <Path d="M 24,5 L 30,7 L 24,9 Z" fill={STROKE} stroke="none" />
              <Path d="M 4,14 Q 10,20 16,14" />
            </G>

            {/* Feather */}
            <G transform="translate(35,150) rotate(18)">
              <Path d="M 0,0 Q 8,-25 0,-55 Q -8,-25 0,0 Z" />
              <Line x1="0" y1="-4" x2="0" y2="-50" />
            </G>

            {/* Binoculars */}
            <G transform="translate(135,130)">
              <Circle cx="0" cy="10" r="9" />
              <Circle cx="22" cy="10" r="9" />
              <Line x1="8" y1="5" x2="14" y2="5" />
            </G>

            {/* Nest with eggs */}
            <G transform="translate(75,88)">
              <Path d="M -13,0 Q -13,11 0,11 Q 13,11 13,0" />
              <Ellipse cx="-6" cy="2" rx="3.2" ry="2.4" />
              <Ellipse cx="0" cy="0" rx="3.2" ry="2.4" />
              <Ellipse cx="6" cy="2" rx="3.2" ry="2.4" />
            </G>

            {/* Scattered dots */}
            <Circle cx="95" cy="55" r="1.2" fill={STROKE} stroke="none" />
            <Circle cx="165" cy="75" r="1.1" fill={STROKE} stroke="none" />
            <Circle cx="55" cy="178" r="1.3" fill={STROKE} stroke="none" />
            <Circle cx="14" cy="105" r="1" fill={STROKE} stroke="none" />
            <Circle cx="150" cy="45" r="1.2" fill={STROKE} stroke="none" />
            <Circle cx="185" cy="150" r="1.1" fill={STROKE} stroke="none" />
          </G>
        </Pattern>
      </Defs>
      <Rect x="0" y="0" width="100%" height="100%" fill="url(#birdDoodles)" />
    </Svg>
  );
}
