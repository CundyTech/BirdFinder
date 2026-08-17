import React, { useEffect, useState } from 'react';
import { StatusBar } from 'react-native';
import { Provider, useDispatch, useSelector } from 'react-redux';
import { store } from './src/store';
import { hydrateLifeList } from './src/store/lifeListSlice';
import { hydrateFilm, claimDailyRefill, claimTrophyRewards } from './src/store/filmSlice';
import { hydratePremium } from './src/store/premiumSlice';
import useTrophyCategories from './src/hooks/useTrophyCategories';
import styles from './src/styles';
import HomeScreen from './src/screens/HomeScreen';
import LifeListScreen from './src/screens/LifeListScreen';
import SpeciesGalleryScreen from './src/screens/SpeciesGalleryScreen';
import TrophyCabinetScreen from './src/screens/TrophyCabinetScreen';

// No routing library — a plain navigation stack of {name, ...params}
// covers this app's whole (still small) navigation surface. A stack
// (rather than one screen slot) is needed because the species gallery is
// now reachable from two different places (the life list and the trophy
// cabinet), so "back" has to return to whichever one you came from.
function RootNavigator() {
  const dispatch = useDispatch();
  const [stack, setStack] = useState([{ name: 'home' }]);
  const screen = stack[stack.length - 1];
  const filmHydrated = useSelector((state) => state.film.hydrated);
  const trophyCategories = useTrophyCategories();

  useEffect(() => {
    dispatch(hydrateLifeList());
    dispatch(hydrateFilm());
    dispatch(hydratePremium());
  }, [dispatch]);

  // Daily free Film — safe to dispatch on every launch, the storage layer
  // no-ops if today's already been claimed.
  useEffect(() => {
    if (filmHydrated) dispatch(claimDailyRefill());
  }, [filmHydrated, dispatch]);

  // Trophy payouts — recomputed (and re-dispatched) whenever trophy state
  // changes; already-claimed trophies are filtered out in the storage
  // layer, so passing the full unlocked list every time is intentional,
  // not a bug.
  useEffect(() => {
    if (!filmHydrated) return;
    const unlockedKeys = trophyCategories.flatMap((category) =>
      (category.trophies || [])
        .filter((trophy) => trophy.unlocked)
        .map((trophy) => `${category.id}:${trophy.label}`)
    );
    if (unlockedKeys.length > 0) dispatch(claimTrophyRewards(unlockedKeys));
  }, [filmHydrated, trophyCategories, dispatch]);

  const push = (next) => setStack((s) => [...s, next]);
  const pop = () => setStack((s) => (s.length > 1 ? s.slice(0, -1) : s));

  let activeScreen;
  if (screen.name === 'lifelist') {
    activeScreen = (
      <LifeListScreen
        onBack={pop}
        onOpenSpecies={(speciesId) => push({ name: 'species', speciesId })}
      />
    );
  } else if (screen.name === 'trophies') {
    activeScreen = (
      <TrophyCabinetScreen
        onBack={pop}
        onOpenSpecies={(speciesId) => push({ name: 'species', speciesId })}
      />
    );
  } else if (screen.name === 'species') {
    activeScreen = <SpeciesGalleryScreen speciesId={screen.speciesId} onBack={pop} />;
  } else {
    activeScreen = (
      <HomeScreen
        onOpenLifeList={() => push({ name: 'lifelist' })}
        onOpenTrophies={() => push({ name: 'trophies' })}
      />
    );
  }

  return (
    <>
      {/* Rendered once here (not per-screen) — React Native's StatusBar
          reverts to the OS default the moment whichever component set it
          unmounts, so setting it per-screen made it flash back to a plain
          white/default bar on every navigation. An opaque backgroundColor
          matching the app's dark theme (not 'transparent') is needed too —
          transparent just reveals the Android window's default light
          background behind the status bar icons. */}
      <StatusBar barStyle="light-content" backgroundColor={styles.PALETTE.bg} />
      {activeScreen}
    </>
  );
}

export default function App() {
  return (
    <Provider store={store}>
      <RootNavigator />
    </Provider>
  );
}
