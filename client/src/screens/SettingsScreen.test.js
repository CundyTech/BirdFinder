import React from 'react';
import renderer, { act } from 'react-test-renderer';
import { Text, TouchableOpacity, Alert, DevSettings } from 'react-native';
import { Provider } from 'react-redux';
import { configureStore } from '@reduxjs/toolkit';
import AsyncStorage from '@react-native-async-storage/async-storage';
import SettingsScreen from './SettingsScreen';
import settingsReducer, { hydrateSettings } from '../store/settingsSlice';
import rewardsReducer, { hydrateRewards } from '../store/rewardsSlice';
import useTrophyCategories from '../hooks/useTrophyCategories';

// useTrophyCategories does a network-backed rarity fetch and reads several
// redux slices — mocking it isolates the debug section's own logic from all
// of that (same reasoning as PerksModal.test.js, which this suite's debug
// tests were moved from).
jest.mock('../hooks/useTrophyCategories');

jest.mock('@expo/vector-icons', () => ({ Feather: 'Feather' }));

const MILESTONES = [
  { key: 'First Reroll', label: 'First Reroll', description: 'Save 10 sightings to earn a free reroll token.', reward: { type: 'reroll', amount: 1 } },
  { key: 'Frequent Flyer', label: 'Frequent Flyer', description: 'Save 25 sightings to unlock the Floral Frame.', reward: { type: 'frame', frameId: 'floral-frame' } },
  { key: 'Century Club', label: 'Century Club', description: 'Save 50 sightings to unlock the Polaroid Frame.', reward: { type: 'frame', frameId: 'polaroid-frame' } },
  { key: 'Locked In', label: 'Locked In', description: 'Reach a 7-day streak to earn a streak protection token.', reward: { type: 'streakProtection', amount: 1 } },
  { key: 'Dedicated Birder', label: 'Dedicated Birder', description: 'Reach a 30-day streak to unlock the Wood Frame.', reward: { type: 'frame', frameId: 'wood-frame' } },
  { key: 'Field Notes', label: 'Field Notes', description: 'Log a sighting on 30 different days to unlock the Calendar Frame.', reward: { type: 'frame', frameId: 'calendar-frame' } },
  { key: "Completionist's Notes", label: "Completionist's Notes", description: 'Complete every Rarity trophy to unlock deep-dive lore.', reward: { type: 'deepDiveLore' } },
  { key: 'Full Cabinet', label: 'Full Cabinet', description: 'Unlock every other trophy to earn the Baroque Frame.', reward: { type: 'frame', frameId: 'baroque-frame' } },
];

function makeStore({ theme = 'system', rewardsOverrides = {} } = {}) {
  const store = configureStore({ reducer: { settings: settingsReducer, rewards: rewardsReducer } });
  store.dispatch(hydrateSettings.fulfilled({ theme }));
  store.dispatch(
    hydrateRewards.fulfilled({
      rerollTokens: 0,
      streakProtectionTokens: 0,
      ownedFrameIds: [],
      equippedFrameId: null,
      deepDiveLoreUnlocked: false,
      claimedMilestoneKeys: [],
      ...rewardsOverrides,
    })
  );
  return store;
}

function renderScreen(store) {
  let tree;
  act(() => {
    tree = renderer.create(
      <Provider store={store}>
        <SettingsScreen onBack={() => {}} />
      </Provider>
    );
  });
  return tree.root;
}

function textOf(node) {
  return Array.isArray(node.props.children) ? node.props.children.join('') : node.props.children;
}

function findByText(root, text) {
  return root.findAll((node) => node.type === Text && textOf(node) === text);
}

function pressRowNamed(root, title) {
  let node = findByText(root, title)[0].parent;
  while (node && node.type !== TouchableOpacity) node = node.parent;
  if (!node) throw new Error(`"${title}" isn't inside a pressable row`);
  act(() => {
    node.props.onPress();
  });
}

describe('SettingsScreen', () => {
  beforeEach(() => {
    useTrophyCategories.mockReturnValue([{ id: 'milestones', trophies: MILESTONES }]);
  });

  it('offers System, Light and Dark appearance options', () => {
    const root = renderScreen(makeStore());
    expect(findByText(root, 'System')).toHaveLength(1);
    expect(findByText(root, 'Light')).toHaveLength(1);
    expect(findByText(root, 'Dark')).toHaveLength(1);
  });

  it('picking a theme updates and persists the preference', async () => {
    const store = makeStore({ theme: 'system' });
    const root = renderScreen(store);

    await act(async () => {
      pressRowNamed(root, 'Dark');
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(store.getState().settings.theme).toBe('dark');
  });

  it('shows the debug tools in a __DEV__ build', () => {
    const root = renderScreen(makeStore());
    expect(findByText(root, 'Unlock All Perks')).toHaveLength(1);
    expect(findByText(root, 'Wipe All Data')).toHaveLength(1);
  });

  it('hides the debug tools outside a __DEV__ build', () => {
    const originalDev = global.__DEV__;
    global.__DEV__ = false;
    try {
      const root = renderScreen(makeStore());
      expect(findByText(root, 'Unlock All Perks')).toHaveLength(0);
      expect(findByText(root, 'Wipe All Data')).toHaveLength(0);
    } finally {
      global.__DEV__ = originalDev;
    }
  });

  it('Unlock All Perks grants every milestone reward at once, regardless of live progress', async () => {
    const store = makeStore();
    const root = renderScreen(store);

    await act(async () => {
      pressRowNamed(root, 'Unlock All Perks');
      // Flush the grantMilestoneRewards thunk's storage round-trip.
      await Promise.resolve();
      await Promise.resolve();
    });

    const state = store.getState().rewards;
    expect(new Set(state.ownedFrameIds)).toEqual(
      new Set(['floral-frame', 'polaroid-frame', 'wood-frame', 'calendar-frame', 'baroque-frame'])
    );
    expect(state.rerollTokens).toBe(1);
    expect(state.streakProtectionTokens).toBe(1);
    expect(state.deepDiveLoreUnlocked).toBe(true);
    expect(state.claimedMilestoneKeys).toHaveLength(MILESTONES.length);
  });

  it('Wipe All Data confirms before clearing storage and reloading', async () => {
    const alertSpy = jest.spyOn(Alert, 'alert');
    const reloadSpy = jest.spyOn(DevSettings, 'reload').mockImplementation(() => {});
    const clearSpy = jest.spyOn(AsyncStorage, 'clear').mockResolvedValue();

    const root = renderScreen(makeStore());

    act(() => {
      pressRowNamed(root, 'Wipe All Data');
    });

    expect(alertSpy).toHaveBeenCalledTimes(1);
    expect(clearSpy).not.toHaveBeenCalled();

    const buttons = alertSpy.mock.calls[0][2];
    const wipeButton = buttons.find((b) => b.text === 'Wipe');
    expect(wipeButton.style).toBe('destructive');

    await act(async () => {
      await wipeButton.onPress();
    });

    expect(clearSpy).toHaveBeenCalledTimes(1);
    expect(reloadSpy).toHaveBeenCalledTimes(1);

    alertSpy.mockRestore();
    reloadSpy.mockRestore();
    clearSpy.mockRestore();
  });
});
