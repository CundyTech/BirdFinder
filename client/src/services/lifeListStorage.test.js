import AsyncStorage from '@react-native-async-storage/async-storage';
import * as FileSystem from 'expo-file-system';
import { loadSightings, addSighting, removeSighting } from './lifeListStorage';

// expo-file-system has no official jest mock (unlike async-storage) — this
// only needs to satisfy the calls lifeListStorage.js actually makes:
// existence checks, directory creation, copy, delete. No real file I/O.
jest.mock('expo-file-system', () => ({
  documentDirectory: 'file:///mock-documents/',
  getInfoAsync: jest.fn(),
  makeDirectoryAsync: jest.fn(),
  copyAsync: jest.fn(),
  deleteAsync: jest.fn(),
}));

describe('lifeListStorage', () => {
  beforeEach(async () => {
    await AsyncStorage.clear();
    jest.clearAllMocks();
    FileSystem.getInfoAsync.mockResolvedValue({ exists: true });
    FileSystem.copyAsync.mockResolvedValue();
    FileSystem.deleteAsync.mockResolvedValue();
    FileSystem.makeDirectoryAsync.mockResolvedValue();
  });

  describe('loadSightings', () => {
    it('returns an empty array when nothing has been saved', async () => {
      expect(await loadSightings()).toEqual([]);
    });

    it('recovers from corrupted JSON instead of throwing', async () => {
      await AsyncStorage.setItem('lifeList.sightings.v1', 'not json{');
      expect(await loadSightings()).toEqual([]);
    });

    it('recovers from a non-array stored value', async () => {
      await AsyncStorage.setItem('lifeList.sightings.v1', JSON.stringify({ not: 'an array' }));
      expect(await loadSightings()).toEqual([]);
    });
  });

  describe('addSighting', () => {
    it('creates the sightings directory only if it does not already exist', async () => {
      FileSystem.getInfoAsync.mockResolvedValue({ exists: false });

      await addSighting({ speciesId: 'Barn_Owl', confidence: 0.95, sourceUri: 'file:///cache/photo.jpg' });

      expect(FileSystem.makeDirectoryAsync).toHaveBeenCalledWith(
        expect.stringContaining('sightings/'),
        { intermediates: true }
      );
    });

    it('skips directory creation when it already exists', async () => {
      FileSystem.getInfoAsync.mockResolvedValue({ exists: true });

      await addSighting({ speciesId: 'Barn_Owl', confidence: 0.95, sourceUri: 'file:///cache/photo.jpg' });

      expect(FileSystem.makeDirectoryAsync).not.toHaveBeenCalled();
    });

    it('copies the photo into permanent storage and records the new sighting', async () => {
      const updated = await addSighting({
        speciesId: 'Barn_Owl',
        confidence: 0.95,
        sourceUri: 'file:///cache/photo.jpg',
      });

      expect(FileSystem.copyAsync).toHaveBeenCalledTimes(1);
      const { from, to } = FileSystem.copyAsync.mock.calls[0][0];
      expect(from).toBe('file:///cache/photo.jpg');
      expect(to).toMatch(/\.jpg$/);

      expect(updated).toHaveLength(1);
      expect(updated[0]).toMatchObject({ speciesId: 'Barn_Owl', confidence: 0.95, photoUri: to });
      expect(updated[0].id).toBeTruthy();
      expect(typeof updated[0].capturedAt).toBe('string');
    });

    it('preserves the source file extension', async () => {
      const updated = await addSighting({ speciesId: 'Barn_Owl', confidence: 0.9, sourceUri: 'file:///cache/photo.png' });
      expect(updated[0].photoUri).toMatch(/\.png$/);
    });

    it('falls back to .jpg when the source has no recognizable extension', async () => {
      const updated = await addSighting({ speciesId: 'Barn_Owl', confidence: 0.9, sourceUri: 'file:///cache/photo' });
      expect(updated[0].photoUri).toMatch(/\.jpg$/);
    });

    it('prepends new sightings so the most recent is first', async () => {
      await addSighting({ speciesId: 'Barn_Owl', confidence: 0.9, sourceUri: 'file:///a.jpg' });
      const updated = await addSighting({ speciesId: 'Mallard', confidence: 0.8, sourceUri: 'file:///b.jpg' });

      expect(updated.map((s) => s.speciesId)).toEqual(['Mallard', 'Barn_Owl']);
    });
  });

  describe('removeSighting', () => {
    it('removes the matching sighting and best-effort deletes its photo file', async () => {
      const [sighting] = await addSighting({ speciesId: 'Barn_Owl', confidence: 0.9, sourceUri: 'file:///a.jpg' });

      const remaining = await removeSighting(sighting.id);

      expect(remaining).toEqual([]);
      expect(FileSystem.deleteAsync).toHaveBeenCalledWith(sighting.photoUri, { idempotent: true });
    });

    it('leaves the list untouched and skips the file delete when the id does not exist', async () => {
      await addSighting({ speciesId: 'Barn_Owl', confidence: 0.9, sourceUri: 'file:///a.jpg' });

      const result = await removeSighting('does-not-exist');

      expect(result).toHaveLength(1);
      expect(FileSystem.deleteAsync).not.toHaveBeenCalled();
    });
  });
});
