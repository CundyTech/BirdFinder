import AsyncStorage from '@react-native-async-storage/async-storage';
import * as FileSystem from 'expo-file-system';

const STORAGE_KEY = 'lifeList.sightings.v1';
const SIGHTINGS_DIR = `${FileSystem.documentDirectory}sightings/`;

async function ensureSightingsDir() {
  const info = await FileSystem.getInfoAsync(SIGHTINGS_DIR);
  if (!info.exists) {
    await FileSystem.makeDirectoryAsync(SIGHTINGS_DIR, { intermediates: true });
  }
}

export async function loadSightings() {
  const raw = await AsyncStorage.getItem(STORAGE_KEY);
  if (!raw) return [];
  try {
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

async function saveSightings(sightings) {
  await AsyncStorage.setItem(STORAGE_KEY, JSON.stringify(sightings));
}

// Copies the photo out of ImagePicker's temp cache into permanent app
// storage (the cache can be cleared by the OS at any time), then appends a
// sighting record and persists the updated list.
export async function addSighting({ speciesId, confidence, sourceUri }) {
  await ensureSightingsDir();

  const id = `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
  const extMatch = /(\.[0-9a-z]+)$/i.exec(sourceUri.split('/').pop() || '');
  const ext = extMatch ? extMatch[1] : '.jpg';
  const destUri = `${SIGHTINGS_DIR}${id}${ext}`;

  await FileSystem.copyAsync({ from: sourceUri, to: destUri });

  const sighting = {
    id,
    speciesId,
    confidence,
    photoUri: destUri,
    capturedAt: new Date().toISOString(),
  };

  const sightings = await loadSightings();
  const updated = [sighting, ...sightings];
  await saveSightings(updated);
  return updated;
}

export async function removeSighting(id) {
  const sightings = await loadSightings();
  const target = sightings.find((s) => s.id === id);
  const updated = sightings.filter((s) => s.id !== id);
  await saveSightings(updated);

  if (target) {
    // Best-effort: an already-missing file shouldn't block removing the record.
    await FileSystem.deleteAsync(target.photoUri, { idempotent: true });
  }

  return updated;
}
