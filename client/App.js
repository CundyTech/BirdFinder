import React, { useEffect, useState } from 'react';
import { Provider, useDispatch } from 'react-redux';
import { store } from './src/store';
import { hydrateLifeList } from './src/store/lifeListSlice';
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

  if (screen.name === 'lifelist') {
    return (
      <LifeListScreen
        onBack={pop}
        onOpenSpecies={(speciesId) => push({ name: 'species', speciesId })}
      />
    );
  }

  if (screen.name === 'trophies') {
    return (
      <TrophyCabinetScreen
        onBack={pop}
        onOpenSpecies={(speciesId) => push({ name: 'species', speciesId })}
      />
    );
  }

  if (screen.name === 'species') {
    return <SpeciesGalleryScreen speciesId={screen.speciesId} onBack={pop} />;
  }

  return (
    <HomeScreen
      onOpenLifeList={() => push({ name: 'lifelist' })}
      onOpenTrophies={() => push({ name: 'trophies' })}
    />
  );
}

export default function App() {
  return (
    <Provider store={store}>
      <RootNavigator />
    </Provider>
  );
}
