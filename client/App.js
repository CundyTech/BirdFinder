import React, { useEffect, useState } from 'react';
import { Provider, useDispatch } from 'react-redux';
import { store } from './src/store';
import { hydrateLifeList } from './src/store/lifeListSlice';
import HomeScreen from './src/screens/HomeScreen';
import LifeListScreen from './src/screens/LifeListScreen';
import SpeciesGalleryScreen from './src/screens/SpeciesGalleryScreen';

// No routing library — the app is small enough that a single piece of
// screen state (with an optional param) covers the whole navigation surface.
function RootNavigator() {
  const dispatch = useDispatch();
  const [screen, setScreen] = useState({ name: 'home' });

  useEffect(() => {
    dispatch(hydrateLifeList());
  }, [dispatch]);

  if (screen.name === 'lifelist') {
    return (
      <LifeListScreen
        onBack={() => setScreen({ name: 'home' })}
        onOpenSpecies={(speciesId) => setScreen({ name: 'species', speciesId })}
      />
    );
  }

  if (screen.name === 'species') {
    return (
      <SpeciesGalleryScreen
        speciesId={screen.speciesId}
        onBack={() => setScreen({ name: 'lifelist' })}
      />
    );
  }

  return <HomeScreen onOpenLifeList={() => setScreen({ name: 'lifelist' })} />;
}

export default function App() {
  return (
    <Provider store={store}>
      <RootNavigator />
    </Provider>
  );
}
