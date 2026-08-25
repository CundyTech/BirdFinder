const React = require('react');

// react-native-svg-transformer turns a .svg import into a React component at
// build time via a Babel/Metro transform Jest doesn't run — this stands in
// for that component in tests so importing one doesn't crash the resolver.
module.exports = React.forwardRef((props, ref) => React.createElement('SvgMock', { ...props, ref }));
