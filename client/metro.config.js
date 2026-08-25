const { getDefaultConfig } = require('expo/metro-config');

// Lets .svg files be imported directly as React components (used for the
// hand-drawn cosmetic photo frames in assets/frames/) instead of needing to
// embed their markup as JS strings.
const config = getDefaultConfig(__dirname);

config.transformer.babelTransformerPath = require.resolve('react-native-svg-transformer');
config.resolver.assetExts = config.resolver.assetExts.filter((ext) => ext !== 'svg');
config.resolver.sourceExts = [...config.resolver.sourceExts, 'svg'];

module.exports = config;
