const math = require('math');


// utils
function splitArray(array, part) {
    var tmp = [];
    for(var i = 0; i < array.length; i += part) {
        tmp.push(array.slice(i, i + part));
    }
    return tmp;
}

function dot(A, B) {
    var result = new Array(A.length).fill(0).map(row => new Array(B[0].length).fill(0));

    return result.map((row, i) => {
        return row.map((val, j) => {
            return A[i].reduce((sum, elm, k) => sum + (elm*B[k][j]) ,0)
        })
    })
}


transpose = m => m[0].map((x,i) => m.map(x => x[i]))

const weights = [splitArray(Array(15).fill(1).map(x=>Math.random()), 3),
                 splitArray(Array(15).fill(1).map(x=>Math.random()), 5),
                 splitArray(Array(5).fill(1).map(x=>Math.random()), 1),
                ];
                 //Array(5).fill(1).map(x=>Math.random()),

let input = [[1.2, 3.4, 1.4]];

function forward(x, weights) {
  for (let i = 0; i < weights.length - 1; i++) {
    x = dot(x, transpose(weights[i]))
    
  }

  return x[0]
}

out = forward(input, weights)

console.log(out);
