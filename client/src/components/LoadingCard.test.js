import React from 'react';
import renderer, { act } from 'react-test-renderer';
import { Text } from 'react-native';
import { Provider } from 'react-redux';
import { configureStore } from '@reduxjs/toolkit';
import LoadingCard from './LoadingCard';
import settingsReducer from '../store/settingsSlice';

// The real Icon component reads expo-font's native module state, which
// isn't set up under Jest and throws on render (see PerksModal.test.js for
// the same issue) — a plain placeholder sidesteps it.
jest.mock('@expo/vector-icons', () => ({ MaterialCommunityIcons: 'MaterialCommunityIcons' }));

// useAppStyles (styles.js) reads the theme preference from the settings
// slice, so any component using it needs a Provider now, even one with
// nothing else to do with Redux.
const store = configureStore({ reducer: { settings: settingsReducer } });

function withStore(ui) {
  return <Provider store={store}>{ui}</Provider>;
}

function textOf(component) {
  return component.root
    .findAllByType(Text)
    .map((node) => Array.isArray(node.props.children) ? node.props.children.join('') : node.props.children)
    .join(' ');
}

describe('LoadingCard', () => {
  it('shows a bird-recognition loading status', () => {
    let component;

    act(() => {
      component = renderer.create(withStore(<LoadingCard />));
    });

    const textContent = textOf(component);
    expect(textContent).toContain('Scanning image');
    expect(textContent).toContain('Matching feather patterns');
  });

  it('calls out that this identification will spend a Film token by default', () => {
    let component;

    act(() => {
      component = renderer.create(withStore(<LoadingCard spendKind="film" />));
    });

    expect(textOf(component)).toContain('Using 1 Film for this ID');
  });

  it('calls out a reroll token instead once Film is at zero', () => {
    let component;

    act(() => {
      component = renderer.create(withStore(<LoadingCard spendKind="reroll" />));
    });

    expect(textOf(component)).toContain('Using 1 reroll token for this ID');
  });

  it('calls out that an ad covered this one for free, not a reroll or Film', () => {
    let component;

    act(() => {
      component = renderer.create(withStore(<LoadingCard spendKind="free-ad" />));
    });

    const textContent = textOf(component);
    expect(textContent).toContain('Free — covered by your ad');
    expect(textContent).not.toContain('Using 1 Film');
    expect(textContent).not.toContain('reroll');
  });

  it('shows no cost at all once unlocked forever', () => {
    let component;

    act(() => {
      component = renderer.create(withStore(<LoadingCard spendKind="unlimited" />));
    });

    const textContent = textOf(component);
    expect(textContent).not.toContain('Film');
    expect(textContent).not.toContain('reroll');
  });
});
