import { configureStore } from '@reduxjs/toolkit';
import lifeListReducer, { hydrateLifeList, recordSighting, deleteSighting } from './lifeListSlice';
import * as lifeListStorage from '../services/lifeListStorage';

// Storage itself is covered by lifeListStorage.test.js — mocking it here
// isolates the reducer/thunk wiring, which is what this file is actually
// testing.
jest.mock('../services/lifeListStorage');

function makeStore() {
  return configureStore({ reducer: { lifeList: lifeListReducer } });
}

describe('lifeListSlice', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('starts empty and unhydrated', () => {
    const store = makeStore();
    expect(store.getState().lifeList).toEqual({ sightings: [], hydrated: false });
  });

  it('hydrateLifeList loads sightings from storage and marks the slice hydrated', async () => {
    const sightings = [{ id: '1', speciesId: 'Barn_Owl' }];
    lifeListStorage.loadSightings.mockResolvedValue(sightings);

    const store = makeStore();
    await store.dispatch(hydrateLifeList());

    expect(store.getState().lifeList).toEqual({ sightings, hydrated: true });
  });

  it('recordSighting forwards its args to storage and replaces state with the result', async () => {
    const updated = [{ id: '2', speciesId: 'Mallard' }];
    lifeListStorage.addSighting.mockResolvedValue(updated);

    const store = makeStore();
    await store.dispatch(
      recordSighting({ speciesId: 'Mallard', confidence: 0.9, sourceUri: 'file:///a.jpg' })
    );

    expect(lifeListStorage.addSighting).toHaveBeenCalledWith({
      speciesId: 'Mallard',
      confidence: 0.9,
      sourceUri: 'file:///a.jpg',
    });
    expect(store.getState().lifeList.sightings).toEqual(updated);
  });

  it('deleteSighting forwards the id to storage and replaces state with the result', async () => {
    lifeListStorage.removeSighting.mockResolvedValue([]);

    const store = makeStore();
    await store.dispatch(deleteSighting('some-id'));

    expect(lifeListStorage.removeSighting).toHaveBeenCalledWith('some-id');
    expect(store.getState().lifeList.sightings).toEqual([]);
  });

  it('does not mark hydrated on a plain recordSighting/deleteSighting — only hydrateLifeList does', async () => {
    lifeListStorage.addSighting.mockResolvedValue([{ id: '1', speciesId: 'Barn_Owl' }]);

    const store = makeStore();
    await store.dispatch(recordSighting({ speciesId: 'Barn_Owl', confidence: 0.9, sourceUri: 'file:///a.jpg' }));

    expect(store.getState().lifeList.hydrated).toBe(false);
  });
});
