const {ReLU, Model} = require('../../../norch');

const model = new Model( { path: 'weights.json', activation: ReLU } );

for (let i = 0; i < 10; i++) {
  let inp = [[Math.round(Math.random() * 100)]];
  let out = model.forward(inp)
  out = Math.round(out);

  if (Math.round(out) === Math.round(inp * 2)) {
    console.log(`Q: ${inp} * 2 | P: ${out} ✅`);
  } else {
    console.log(`Q: ${inp} * 2 | P: ${out} ❌`);
  }
}
