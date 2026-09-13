// Initializes i18next once for the whole test run. Production code always
// gets this via App.js's import graph (every screen imports it directly or
// through a domain module) before anything renders, but a unit test can
// render a leaf component in isolation without pulling that chain in —
// this setup file guarantees the same "already initialized" starting point
// jest-wide, so useTranslation() never needs a real i18next instance provider.
import './src/i18n';
