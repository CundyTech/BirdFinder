import React from 'react';
import { Modal, TouchableOpacity, Image } from 'react-native';
import { Feather } from '@expo/vector-icons';
import styles from '../styles';

export default function ImageLightbox({ uri, label, onClose }) {
  return (
    <Modal visible={Boolean(uri)} transparent animationType="fade" onRequestClose={onClose}>
      <TouchableOpacity style={styles.lightboxBackdrop} activeOpacity={1} onPress={onClose}>
        <TouchableOpacity
          style={styles.lightboxCloseButton}
          onPress={onClose}
          accessibilityRole="button"
          accessibilityLabel="Close image viewer"
          hitSlop={{ top: 12, bottom: 12, left: 12, right: 12 }}
        >
          <Feather name="x" size={24} color="#ffffff" />
        </TouchableOpacity>
        {uri && (
          <Image
            source={{ uri }}
            style={styles.lightboxImage}
            resizeMode="contain"
            accessible
            accessibilityLabel={label || 'Full-screen photo'}
          />
        )}
      </TouchableOpacity>
    </Modal>
  );
}
