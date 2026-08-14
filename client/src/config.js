// Central config for API base URL.
export const API_BASE = 'https://birdfinder-api.cundytech.com';

// Must match the API_KEY env var the server was started with. Set via
// client/.env (gitignored — copy client/.env.example to get started), not
// hardcoded here, so the real value never lands in git history. Still
// extractable by anyone who unpacks the built app — this only stops
// casual/opportunistic traffic, not a determined attacker. See api/README.md.
export const API_KEY = process.env.EXPO_PUBLIC_API_KEY;
if (!API_KEY) {
  console.warn('EXPO_PUBLIC_API_KEY is not set — API requests will be rejected. Copy client/.env.example to client/.env and fill it in.');
}

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
