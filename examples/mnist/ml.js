const { Model, ReLU } = require('../../../norch');

const model = new Model( 'weights.json', ReLU );
const inp = Array(784).fill(0);

console.log(model.forward([inp]))
