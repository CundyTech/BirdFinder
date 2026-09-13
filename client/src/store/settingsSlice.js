import { createAsyncThunk, createSlice } from '@reduxjs/toolkit';
import { loadSettingsState, setTheme as setThemeInStorage } from '../services/settingsStorage';

export const hydrateSettings = createAsyncThunk('settings/hydrate', async () => {
  return loadSettingsState();
});

export const setTheme = createAsyncThunk('settings/setTheme', async (theme) => {
  return setThemeInStorage(theme);
});

const settingsSlice = createSlice({
  name: 'settings',
  initialState: {
    theme: 'system',
    hydrated: false,
  },
  reducers: {},
  extraReducers: (builder) => {
    builder
      .addCase(hydrateSettings.fulfilled, (state, action) => {
        state.theme = action.payload.theme;
        state.hydrated = true;
      })
      .addCase(setTheme.fulfilled, (state, action) => {
        state.theme = action.payload.theme;
      });
  },
});

export default settingsSlice.reducer;
