import React from 'react';
import renderer, { act } from 'react-test-renderer';
import { Text, TouchableOpacity, Alert, DevSettings } from 'react-native';
import { Provider } from 'react-redux';
import { configureStore } from '@reduxjs/toolkit';
import AsyncStorage from '@react-native-async-storage/async-storage';
import PerksModal from './PerksModal';
import rewardsReducer, { hydrateRewards } from '../store/rewardsSlice';
import useTrophyCategories from '../hooks/useTrophyCategories';
import { FRAMES } from '../domain/frames';

// useTrophyCategories does a network-backed rarity fetch and reads several
// redux slices — mocking it isolates PerksModal's own logic (unlocked
// state, the Frames rollup, the debug tools) from all of that.
jest.mock('../hooks/useTrophyCategories');

// @expo/vector-icons' real Icon component reads expo-font's native module
// state (`loadedNativeFonts`), which isn't set up under Jest and throws on
// render — no existing test in this project renders a real icon, so this
// is the first to hit it. Standing in simple placeholders sidesteps that
// without needing to solve icon-font mocking for the whole suite.
jest.mock('@expo/vector-icons', () => ({
  MaterialCommunityIcons: 'MaterialCommunityIcons',
  Feather: 'Feather',
}));

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

function makeStore(rewardsOverrides = {}) {
  const store = configureStore({ reducer: { rewards: rewardsReducer } });
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

function renderModal(store) {
  let tree;
  act(() => {
    tree = renderer.create(
      <Provider store={store}>
        <PerksModal visible onClose={() => {}} />
      </Provider>
    );
  });
  return tree.root;
}

// A <Text> built from multiple interpolated expressions (e.g. `{a} of {b}
// unlocked`) renders `children` as an array of parts, not one string — join
// before comparing, same as LoadingCard.test.js does.
function textOf(node) {
  return Array.isArray(node.props.children) ? node.props.children.join('') : node.props.children;
}

function findByText(root, text) {
  return root.findAll((node) => node.type === Text && textOf(node) === text);
}

// Rows are TouchableOpacity (Frames, debug tools) or a plain View (the
// non-frame perk rows, which have nothing to tap) — walk up from the title
// text to whichever is the nearest interactive ancestor.
function pressRowNamed(root, title) {
  let node = findByText(root, title)[0].parent;
  while (node && node.type !== TouchableOpacity) node = node.parent;
  if (!node) throw new Error(`"${title}" isn't inside a pressable row`);
  act(() => {
    node.props.onPress();
  });
}

describe('PerksModal', () => {
  beforeEach(() => {
    useTrophyCategories.mockReturnValue([{ id: 'milestones', trophies: MILESTONES }]);
  });

  it('shows an unlocked perk with what it does, and a locked one with its requirement', () => {
    const store = makeStore({ claimedMilestoneKeys: ['milestones:First Reroll'] });
    const root = renderModal(store);

    expect(findByText(root, 'Free Reroll')).toHaveLength(1);
    expect(findByText(root, '+1 token — skips the Film cost once each')).toHaveLength(1);

    // Locked In hasn't been claimed, so it shows the requirement text
    // (from the milestone), not rewardActionText.
    expect(findByText(root, 'Reach a 7-day streak to earn a streak protection token.')).toHaveLength(1);
  });

  it('rolls every frame reward up into one Frames row with an accurate count', () => {
    // A stale id left over from a renamed/removed frame shouldn't inflate
    // the count past the number of frames that actually exist.
    const store = makeStore({ ownedFrameIds: ['floral-frame', 'no-longer-a-real-frame'] });
    const root = renderModal(store);

    expect(findByText(root, `1 of ${FRAMES.length} unlocked — tap to choose which one to wear`)).toHaveLength(1);
    // Individual frame milestones (Frequent Flyer, Century Club, ...) don't
    // get their own row.
    expect(findByText(root, 'Floral Frame')).toHaveLength(0);
  });

  it('shows the debug tools in a __DEV__ build', () => {
    const root = renderModal(makeStore());
    expect(findByText(root, 'Unlock All Perks')).toHaveLength(1);
    expect(findByText(root, 'Wipe All Data')).toHaveLength(1);
  });

  it('hides the debug tools outside a __DEV__ build', () => {
    const originalDev = global.__DEV__;
    global.__DEV__ = false;
    try {
      const root = renderModal(makeStore());
      expect(findByText(root, 'Unlock All Perks')).toHaveLength(0);
      expect(findByText(root, 'Wipe All Data')).toHaveLength(0);
    } finally {
      global.__DEV__ = originalDev;
    }
  });

  it('Unlock All Perks grants every milestone reward at once, regardless of live progress', async () => {
    const store = makeStore();
    const root = renderModal(store);

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

    const root = renderModal(makeStore());

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
