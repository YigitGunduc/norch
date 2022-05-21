  const squareSize = 16;
const noRows = 28;
const noCols = 28;

var canvas = document.getElementById('canvas');
if (!canvas.getContext) console.log("error");
var context = canvas.getContext('2d');

displayGridPaper();

let drag = false;

document.addEventListener('mousedown', () => drag = true);
document.addEventListener('mouseup', () => drag = false);
document.addEventListener('mousemove', () => {
  if(drag) {
  let x = window.event.clientX; 
  let y = window.event.clientY; 

  displayCell(x, y)
  }
})

function createMatrix(rows, cols) {
  let mat = [];

  for (let i = 0; i < rows; i++) {
     mat[i] = new Array(cols).fill(0);
  }
  return mat;
}

function displayCell(x, y) {
    var col = Math.floor( x / squareSize );
    var row = Math.floor( y / squareSize );

    if (col >= noCols || row >= noRows) return;

    var cellX = col * squareSize;
    var cellY = row * squareSize;

    console.log(state)
    state[row][col] = 1;

    context.fillRect(cellX, cellY, squareSize, squareSize);
}

function displayGridPaper() {

  for(var row = 0; row < noRows; row++)
  {
      for(var col = 0; col < noCols; col++)
      {
          var x = col * squareSize;
          var y = row * squareSize;

          context.strokeRect(x, y, squareSize, squareSize);
      }
  }
}
