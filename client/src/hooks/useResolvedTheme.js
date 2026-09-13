import { useColorScheme } from 'react-native';
import { useSelector } from 'react-redux';

// Resolves the user's theme preference (settingsSlice — 'system', 'light',
// or 'dark') against the OS setting when it's 'system', to a concrete
// 'light' or 'dark'. Shared by useAppStyles (which stylesheet to build) and
// App.js (which StatusBar style to show) so both always agree.
export default function useResolvedTheme() {
  const themePreference = useSelector((state) => state.settings.theme);
  const systemScheme = useColorScheme();
  return themePreference === 'system' ? systemScheme || 'dark' : themePreference;
}
