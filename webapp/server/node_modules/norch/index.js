const activations = require('./lib/activations');
const norch = require('./lib/norch');

module.exports = {
  activations,
  norch,
  ...activations,
  ...norch
};
