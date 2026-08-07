// Central config for API base URL. Update to your machine IP if needed.
export const API_BASE = 'http://192.168.01.50:8080';

// Below this confidence (%), the low-confidence warning banner shows,
// suggesting a retake.
export const LOW_CONFIDENCE_THRESHOLD = 90;

// Below this stricter confidence (%), the app won't auto-guess a species at
// all — it shows improvement tips and a tappable list of candidates instead
// of presenting a shaky top-1 guess as if it were a confirmed result.
export const UNCERTAIN_THRESHOLD = 30;

// Debug: force species detail to always show for the top guess, bypassing
// the uncertain-result flow above regardless of actual confidence.
export const DEBUG_ALWAYS_SHOW_SPECIES_DETAILS = false;

// Minimum time the loading screen stays up, even if the server responds
// faster — avoids a jarring flash on quick responses.
export const MIN_LOADING_DURATION_MS = 5000;
