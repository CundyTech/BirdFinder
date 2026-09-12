import React from 'react';
import renderer, { act } from 'react-test-renderer';
import { Text, TouchableOpacity } from 'react-native';
import FilmChoiceModal from './FilmChoiceModal';
import { useFilmRewardedAd } from '../hooks/useFilmRewardedAd';

// The real Icon component reads expo-font's native module state, which
// isn't set up under Jest and throws on render (see PerksModal.test.js for
// the same issue) — a plain placeholder sidesteps it.
jest.mock('@expo/vector-icons', () => ({ MaterialCommunityIcons: 'MaterialCommunityIcons' }));

// useFilmRewardedAd wraps react-native-google-mobile-ads, which has no
// native module registered under Jest — even auto-mocking the hook still
// requires the real file first to learn its shape, which pulls in the SDK
// and throws. An explicit factory skips loading the real module entirely.
jest.mock('../hooks/useFilmRewardedAd', () => ({ useFilmRewardedAd: jest.fn() }));

function renderModal(props) {
  let tree;
  act(() => {
    tree = renderer.create(
      <FilmChoiceModal visible onClose={() => {}} filmBalance={3} onUseFilm={() => {}} onAdEarned={() => {}} {...props} />
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

function findRowNamed(root, title) {
  let node = findByText(root, title)[0].parent;
  while (node && node.type !== TouchableOpacity) node = node.parent;
  return node;
}

function pressRowNamed(root, title) {
  act(() => {
    findRowNamed(root, title).props.onPress();
  });
}

describe('FilmChoiceModal', () => {
  beforeEach(() => {
    useFilmRewardedAd.mockReturnValue({ isLoaded: true, showAd: jest.fn() });
  });

  it('offers both Film and an ad, showing the current Film balance', () => {
    const root = renderModal({ filmBalance: 3 });
    expect(findByText(root, 'Use 1 Film')).toHaveLength(1);
    expect(findByText(root, '3 Film available')).toHaveLength(1);
    expect(findByText(root, 'Watch an ad instead')).toHaveLength(1);
    expect(findByText(root, 'Covers this identification, free')).toHaveLength(1);
  });

  it('calls onUseFilm when Use 1 Film is tapped', () => {
    const onUseFilm = jest.fn();
    const root = renderModal({ onUseFilm });

    pressRowNamed(root, 'Use 1 Film');

    expect(onUseFilm).toHaveBeenCalledTimes(1);
  });

  it('closes the modal and shows the ad when Watch an ad instead is tapped', () => {
    const onClose = jest.fn();
    const showAd = jest.fn();
    useFilmRewardedAd.mockReturnValue({ isLoaded: true, showAd });
    const root = renderModal({ onClose });

    pressRowNamed(root, 'Watch an ad instead');

    expect(onClose).toHaveBeenCalledTimes(1);
    expect(showAd).toHaveBeenCalledTimes(1);
  });

  it('disables the ad option while no ad is loaded, without disabling Use 1 Film', () => {
    useFilmRewardedAd.mockReturnValue({ isLoaded: false, showAd: jest.fn() });
    const root = renderModal();

    expect(findByText(root, 'Loading ad...')).toHaveLength(1);

    expect(findRowNamed(root, 'Watch an ad instead').props.disabled).toBe(true);
    expect(findRowNamed(root, 'Use 1 Film').props.disabled).toBeFalsy();
  });

  it('passes onAdEarned straight through to useFilmRewardedAd, so the reward never touches Film', () => {
    const onAdEarned = jest.fn();
    renderModal({ onAdEarned });

    expect(useFilmRewardedAd).toHaveBeenCalledWith(onAdEarned);
  });
});
