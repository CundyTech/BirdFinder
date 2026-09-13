import i18next from 'i18next';
import { initReactI18next } from 'react-i18next';
import * as Localization from 'expo-localization';

import common from './locales/en/common.json';
import home from './locales/en/home.json';
import lifeList from './locales/en/lifeList.json';
import speciesGallery from './locales/en/speciesGallery.json';
import trophyCabinet from './locales/en/trophyCabinet.json';
import components from './locales/en/components.json';
import modals from './locales/en/modals.json';
import rewards from './locales/en/rewards.json';
import groups from './locales/en/groups.json';
import trophyCategories from './locales/en/trophyCategories.json';
import achievements from './locales/en/achievements.json';
import rarityTiers from './locales/en/rarityTiers.json';
import species from './locales/en/species.json';
import birdProfiles from './locales/en/birdProfiles.json';
import deepDive from './locales/en/deepDive.json';
import migrationRoutes from './locales/en/migrationRoutes.json';

// Every language this build ships resources for. To add a language: drop a
// matching set of locales/<code>/*.json files (same keys as locales/en),
// import them above, add a `<code>: { translation: {...} }` entry to
// `resources` below, and list the code here — no other code in the app
// needs to change, since every screen/domain module reads text through
// i18n.t()/useTranslation() rather than hardcoding English.
const SUPPORTED_LANGUAGES = ['en'];

const resources = {
  en: {
    translation: {
      common,
      home,
      lifeList,
      speciesGallery,
      trophyCabinet,
      components,
      modals,
      rewards,
      groups,
      trophyCategories,
      achievements,
      rarityTiers,
      species,
      birdProfiles,
      deepDive,
      migrationRoutes,
    },
  },
};

function detectDeviceLanguage() {
  try {
    const languageCode = Localization.getLocales()?.[0]?.languageCode;
    return languageCode && SUPPORTED_LANGUAGES.includes(languageCode) ? languageCode : 'en';
  } catch {
    return 'en';
  }
}

i18next.use(initReactI18next).init({
  resources,
  lng: detectDeviceLanguage(),
  fallbackLng: 'en',
  supportedLngs: SUPPORTED_LANGUAGES,
  interpolation: { escapeValue: false },
  returnNull: false,
  // React Native has no synchronous-vs-deferred distinction i18next needs to
  // guard against here; disabling it avoids relying on setImmediate.
  initImmediate: false,
});

export default i18next;
