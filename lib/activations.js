'use strict';

module.exports.Identity = function(x) {
    return x;
};

module.exports.Inverse = function(x) {
  if (Array.isArray(x)) {
    return x.map(x => (1 - x));
  }
  return (1 - x);
};

module.exports.Tanh = function(x) {
  if (Array.isArray(x)) {
    return x.map(x => Math.tanh(x));
  }
  return (Math.tanh(x));
};

module.exports.ReLU = function(x) {
  if (Array.isArray(x)) {
    return x.map(x => Math.max(0, x));
  }
  return Math.max(0, x);
};

module.exports.BinaryStep = function(x) {
  if (Array.isArray(x)) {
    return x.map(x => ((x < 0) ? 0 : 1));
  }
  return ((x < 0) ? 0 : 1);
};

module.exports.Logistic = module.exports.Sigmoid = module.exports.SoftStep = function(x) {
  if (Array.isArray(x)) {
    return x.map(x => 1 / (1 + Math.exp(-x)));
  }
  return 1 / (1 + Math.exp(-x));
};

module.exports.SiLU = module.exports.Swish1 = function(x) {
  if (Array.isArray(x)) {
    return x.map(x => x * module.exports.Sigmoid(x));
  }
  return x * module.exports.Sigmoid(x);
};

module.exports.ArcTan = function(x) {
  if (Array.isArray(x)) {
    return x.map(x => Math.atan(x));
  }
  return Math.atan(x);
};

module.exports.GELU = function(x) {
  if (Array.isArray(x)) {
    return x.map(x => (x / 2)*(1+ module.exports.Erf(x / Math.SQRT2)));
  }
  return (x / 2)*(1+ module.exports.Erf(x / Math.SQRT2));
};

module.exports.PReLU = function(x, a) {
  if (Array.isArray(x)) {
    return x.map(x => ((x < 0) ? (a * x) : x));
  }
  return ((x < 0) ? (a * x) : x);
};

module.exports.ELU = function(x, a) {
  if (Array.isArray(x)) {
    return x.map(x => ((x > 0) ? $x : (a*Math.expm1(x))));
  }
  return ((x > 0) ? x : ($a*Math.expm1(x)));
};

module.exports.SELU = function(x) {
  if (Array.isArray(x)) {
    x.map(x => 1.0507 * module.exports.ELU(x, 1.67326));
  }
  return 1.0507 * module.exports.ELU(x, 1.67326);
};

module.exports.Sinusoid = function(x) {
  if (Array.isArray(x)) {
    x.map(x => Math.sin(x));
  }
  return Math.sin(x);
};
