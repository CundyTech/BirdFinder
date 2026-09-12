import React from 'react';
import renderer, { act } from 'react-test-renderer';
import { Text } from 'react-native';
import LoadingCard from './LoadingCard';

describe('LoadingCard', () => {
  it('shows a bird-recognition loading status', () => {
    let component;

    act(() => {
      component = renderer.create(<LoadingCard />);
    });

    const textContent = component.root
      .findAllByType(Text)
      .map((node) => Array.isArray(node.props.children) ? node.props.children.join('') : node.props.children)
      .join(' ');

    expect(textContent).toContain('Scanning image');
    expect(textContent).toContain('Matching feather patterns');
  });
});
