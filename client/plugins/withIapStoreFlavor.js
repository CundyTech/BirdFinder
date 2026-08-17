const { withAppBuildGradle } = require('@expo/config-plugins');

// react-native-iap ships separate Amazon/Play product flavors on Android
// (it supports both app stores). Gradle can't resolve the dependency
// without the app module picking one, since expo prebuild's generated
// build.gradle doesn't know about that flavor dimension.
module.exports = function withIapStoreFlavor(config) {
  return withAppBuildGradle(config, (config) => {
    const marker = "missingDimensionStrategy 'store', 'play'";
    if (config.modResults.contents.includes(marker)) {
      return config;
    }
    config.modResults.contents = config.modResults.contents.replace(
      /defaultConfig\s*{/,
      (match) => `${match}\n        ${marker}`
    );
    return config;
  });
};
