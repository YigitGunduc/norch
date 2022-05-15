function matmul(a, b) {
  var aNumRows = a.length, aNumCols = a[0].length,
  bNumRows = b.length, bNumCols = b[0].length,
  m = new Array(aNumRows);  // initialize array of rows
  for (var r = 0; r < aNumRows; ++r) {
    m[r] = new Array(bNumCols); // initialize the current row
    for (var c = 0; c < bNumCols; ++c) {
      m[r][c] = 0;             // initialize the current cell
      for (var i = 0; i < aNumCols; ++i) {
        m[r][c] += a[r][i] * b[i][c];
      }
    }
  }
  return m;
}

function transpose(m) {
  return m[0].map((x, i) => m.map(x => x[i]));
} 

function add(m, v) {
  for (let i = 0; i < m[0].length; i++) {
    m[0][i] = m[0][i] + v[i];
  }
  return m;
}

module.exports = {
  matmul,
  transpose,
  add
};
