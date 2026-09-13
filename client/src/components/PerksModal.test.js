import React from 'react';
import renderer, { act } from 'react-test-renderer';
import { Text } from 'react-native';
import { Provider } from 'react-redux';
import { configureStore } from '@reduxjs/toolkit';
import PerksModal from './PerksModal';
import rewardsReducer, { hydrateRewards } from '../store/rewardsSlice';
import settingsReducer from '../store/settingsSlice';
import useTrophyCategories from '../hooks/useTrophyCategories';
import { FRAMES } from '../domain/frames';

// useTrophyCategories does a network-backed rarity fetch and reads several
// redux slices — mocking it isolates PerksModal's own logic (unlocked
// state, the Frames rollup) from all of that.
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
  const store = configureStore({ reducer: { rewards: rewardsReducer, settings: settingsReducer } });
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
});
