import Matrix from './matrix.js';

// Activation function: for now use sigmoid function
function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

export default class NeuralNetwork {
  constructor(inputNodes, hiddenNodes, outputNodes) {
    this.inputNodes = inputNodes;
    this.hiddenNodes = hiddenNodes;
    this.outputNodes = outputNodes;

    this.weights_ih = new Matrix(this.hiddenNodes, this.inputNodes); // weights between input and hidden layer
    this.weights_ho = new Matrix(this.outputNodes, this.hiddenNodes); // weights between hidden and output layer
    this.hiddenBias = new Matrix(this.hiddenNodes, 1);
    this.outputBias = new Matrix(this.outputNodes, 1);
    this.weights_ih.randomize();
    this.weights_ho.randomize();
    this.hiddenBias.randomize();
    this.outputBias.randomize();
  }

  feedForward(inputArray) {
    let input = Matrix.fromArray(inputArray);

    let hidden = Matrix.multiply(this.weights_ih, input);
    hidden.add(this.hiddenBias);
    hidden.apply(sigmoid); // apply activation function to the hidden layer

    let output = Matrix.multiply(this.weights_ho, hidden);
    output.add(this.outputBias);
    output.apply(sigmoid); // apply activation function to the output layer

    return output.toArray(); // send the result back to user
  }
}
