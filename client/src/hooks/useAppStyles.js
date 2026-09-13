import { useMemo } from 'react';
import { createStyles } from '../styles';
import useResolvedTheme from './useResolvedTheme';

// Memoizes the stylesheet for the current resolved theme so components only
// rebuild it when the theme actually changes, not on every render.
export default function useAppStyles() {
  const resolvedTheme = useResolvedTheme();
  return useMemo(() => createStyles(resolvedTheme), [resolvedTheme]);
}
