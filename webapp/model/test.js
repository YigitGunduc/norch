const { Model, ReLU } = require('../../../norch');


function createMatrix(rows, cols) {
  let mat = [];

  for (let i = 0; i < rows; i++) {
     mat[i] = new Array(cols).fill(0);
  }
  return mat;
}

inp = createMatrix(16, 16);

const model = new Model( { path: 'weights.json', activation: ReLU } )
let out = model.forward([inp.flat()]);
console.log(out[0])
