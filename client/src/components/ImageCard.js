import React, { useState } from 'react';
import { View, Image, TouchableOpacity, useWindowDimensions } from 'react-native';
import { useTranslation } from 'react-i18next';
import styles from '../styles';
import ImageLightbox from './ImageLightbox';

export default function ImageCard({ uri }) {
  const { t } = useTranslation();
  const [viewerOpen, setViewerOpen] = useState(false);
  const { width, height } = useWindowDimensions();
  const isPortrait = height >= width;
  if (!uri) return null;

  const imageWidth = isPortrait ? Math.min(width - 48, 720) : Math.min(360, width - 120);
  const imageHeight = Math.round(imageWidth * 0.66);

  return (
    <View style={styles.imageCard}>
      <TouchableOpacity
        onPress={() => setViewerOpen(true)}
        accessibilityRole="imagebutton"
        accessibilityLabel={t('common.viewPhotoLarger')}
      >
        <Image source={{ uri }} style={{ width: imageWidth, height: imageHeight, borderRadius: 8 }} />
      </TouchableOpacity>
      <ImageLightbox uri={viewerOpen ? uri : null} label={t('common.photoFullScreen')} onClose={() => setViewerOpen(false)} />
    </View>
  );
}
