import React, { useEffect, useState } from 'react';
import { StatusBar } from 'react-native';
import { Provider, useDispatch } from 'react-redux';
import { store } from './src/store';
import { hydrateLifeList } from './src/store/lifeListSlice';
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

  useEffect(() => {
    dispatch(hydrateLifeList());
  }, [dispatch]);

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
