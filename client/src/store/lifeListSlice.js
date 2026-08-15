import { createAsyncThunk, createSlice } from '@reduxjs/toolkit';
import { loadSightings, addSighting, removeSighting } from '../services/lifeListStorage';

export const hydrateLifeList = createAsyncThunk('lifeList/hydrate', async () => {
  return loadSightings();
});

export const recordSighting = createAsyncThunk(
  'lifeList/recordSighting',
  async ({ speciesId, confidence, sourceUri }) => {
    return addSighting({ speciesId, confidence, sourceUri });
  }
);

export const deleteSighting = createAsyncThunk('lifeList/deleteSighting', async (id) => {
  return removeSighting(id);
});

const lifeListSlice = createSlice({
  name: 'lifeList',
  initialState: {
    sightings: [],
    hydrated: false,
  },
  reducers: {},
  extraReducers: (builder) => {
    builder
      .addCase(hydrateLifeList.fulfilled, (state, action) => {
        state.sightings = action.payload;
        state.hydrated = true;
      })
      .addCase(recordSighting.fulfilled, (state, action) => {
        state.sightings = action.payload;
      })
      .addCase(deleteSighting.fulfilled, (state, action) => {
        state.sightings = action.payload;
      });
  },
});

export default lifeListSlice.reducer;
