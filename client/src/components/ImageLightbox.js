import React from 'react';
import { Modal, TouchableOpacity, View, Image, useWindowDimensions } from 'react-native';
import { Feather } from '@expo/vector-icons';
import { useSelector } from 'react-redux';
import { useTranslation } from 'react-i18next';
import styles from '../styles';
import { getFrame } from '../domain/frames';
import FrameBorder from './FrameBorder';

const MAX_WIDTH_FRACTION = 0.92;
const MAX_HEIGHT_FRACTION = 0.75;

// Fits a box of `aspectRatio` (width/height) within maxWidth x maxHeight,
// preferring the widest box that doesn't overflow either bound — the same
// "contain" logic CSS's object-fit does, computed in JS since Yoga's
// aspectRatio style needs at least one definite dimension to seed from,
// which an absolutely-positioned-children box doesn't have.
function fitBox(aspectRatio, maxWidth, maxHeight) {
  let width = maxWidth;
  let height = width / aspectRatio;
  if (height > maxHeight) {
    height = maxHeight;
    width = height * aspectRatio;
  }
  return { width, height };
}

// The one place a cosmetic frame is shown around a photo itself (as
// opposed to the small ring around the header logo) — deliberately kept
// off grid thumbnails, which stay uncluttered. When a frame is equipped,
// the photo is cropped (resizeMode="cover") to exactly fill the frame's
// hand-drawn window cutout rather than just placed behind an unrelated
// border, so it reads as a real photo sitting inside the frame.
export default function ImageLightbox({ uri, label, onClose }) {
  const { t } = useTranslation();
  const equippedFrame = useSelector((state) => getFrame(state.rewards.equippedFrameId));
  const { width: screenWidth, height: screenHeight } = useWindowDimensions();

  let framedBox = null;
  let windowStyle = null;
  if (equippedFrame) {
    const { viewBox, window } = equippedFrame;
    framedBox = fitBox(viewBox.width / viewBox.height, screenWidth * MAX_WIDTH_FRACTION, screenHeight * MAX_HEIGHT_FRACTION);
    windowStyle = {
      left: `${(window.x / viewBox.width) * 100}%`,
      top: `${(window.y / viewBox.height) * 100}%`,
      width: `${(window.width / viewBox.width) * 100}%`,
      height: `${(window.height / viewBox.height) * 100}%`,
    };
  }

  return (
    <Modal visible={Boolean(uri)} transparent animationType="fade" onRequestClose={onClose}>
      <TouchableOpacity style={styles.lightboxBackdrop} activeOpacity={1} onPress={onClose}>
        <TouchableOpacity
          style={styles.lightboxCloseButton}
          onPress={onClose}
          accessibilityRole="button"
          accessibilityLabel={t('components.imageLightbox.closeImageViewer')}
          hitSlop={{ top: 12, bottom: 12, left: 12, right: 12 }}
        >
          <Feather name="x" size={24} color="#ffffff" />
        </TouchableOpacity>
        {uri && equippedFrame && (
          <View style={[styles.framedPhotoBox, framedBox]}>
            <Image
              source={{ uri }}
              style={[styles.framedPhotoImage, windowStyle]}
              resizeMode="cover"
              accessible
              accessibilityLabel={label || t('components.imageLightbox.fullScreenPhoto')}
            />
            <FrameBorder frame={equippedFrame} />
          </View>
        )}
        {uri && !equippedFrame && (
          <Image
            source={{ uri }}
            style={styles.lightboxImage}
            resizeMode="contain"
            accessible
            accessibilityLabel={label || t('components.imageLightbox.fullScreenPhoto')}
          />
        )}
      </TouchableOpacity>
    </Modal>
  );
}
