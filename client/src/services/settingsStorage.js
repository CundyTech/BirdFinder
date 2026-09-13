import AsyncStorage from '@react-native-async-storage/async-storage';

const STORAGE_KEY = 'settings.state.v1';
const VALID_THEMES = ['system', 'light', 'dark'];

export async function loadSettingsState() {
  const raw = await AsyncStorage.getItem(STORAGE_KEY);
  if (!raw) return { theme: 'system' };
  try {
    const parsed = JSON.parse(raw);
    return { theme: VALID_THEMES.includes(parsed?.theme) ? parsed.theme : 'system' };
  } catch {
    return { theme: 'system' };
  }
}

export async function setTheme(theme) {
  const state = { theme: VALID_THEMES.includes(theme) ? theme : 'system' };
  await AsyncStorage.setItem(STORAGE_KEY, JSON.stringify(state));
  return state;
}
