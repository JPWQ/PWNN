#include "main.h"

Layer createLayer(int prevSize, int curSize, bool inputLayer) {
  // float weightsArr[curSize][prevSize];

  // The weights 2d array takes n float pointers where n is the number of neurons in the next/current layer.
  float** weightsArr = (float**)malloc(sizeof(float*) * curSize);

  // Each neruon has it's own bias so it must be a 1d array
  // taking the size of the number of neurons in the next/current layer.
  // I chose this because after forward and backward propagation each neruon
  // should be better optimized.
  float* biasArr = malloc(sizeof(float) * curSize);

  // Intialize the output array of the current layer.
  // This will be overwritten by the Compute Layer function.
  float* outputArr = malloc(sizeof(float) * prevSize);
  
  // For each neuron in the next/current layer initialize
  // each element of output array is a random number greater than 0 if the curLayer is not an input layer
  // each element of the weights array is a float pointer of size of the previous layer
  // each element of the biass array is a random number greater than 0

  // For each neuron in the previous layer initialize
  // the ith pointer to a random number greater than 0
  for(int i = 0; i < curSize; i++) {
    if(!inputLayer) {
      outputArr[i] = rand() + 1;
    }

    if(prevSize > 0) {
      weightsArr[i] = malloc(sizeof(float) * prevSize);
    }

    biasArr[i] = rand() + 1;

    for(int j = 0; j < prevSize; j++) {
      weightsArr[i][j] = rand() + 1;
    }
  }

  Layer newLayer = {weightsArr, biasArr, outputArr, curSize, inputLayer};
  return newLayer;
}

// Using the weights and bias from the previous layer compute the the output of the current layer
void computeLayer(Layer* prevLayer, Layer* curLayer, bool inputLayer) {
  // If previous layer pointer is null then the current layer is the input layer
  if (inputLayer) {
    for(int i = 0; i < curLayer->curSize; i++) {
      curLayer->output[i] = rand() + 1;
    }
    return;
  }

  // Using the formula Z = W*A + B

  // Taking the transpose is unnecessary because
  // the weights matrix uses the row vector convention

  // Then using the sigmoid activation function apply it to each output in the output vector
  float* result = matrixMult(prevLayer->weights, prevLayer->output, curLayer->curSize, prevLayer->curSize);
  for(int i = 0; i < curLayer->curSize; i++) {
    result[i] += prevLayer->bias[i];
    result[i] = sigmoidf(result[i]);
  }
  curLayer->output = result;
}

void freeL(Layer layer) {
  for(int i = 0; i < layer.curSize; i++) {
    free(layer.weights[i]);
  }
  free(layer.weights);
  free(layer.bias);
  free(layer.output);
}

Layer* createNeuralNetwork(int numLayers, ...) {
  Layer* neuralNetwork = malloc(sizeof(Layer) * numLayers);

  va_list args;
  va_start(args, numLayers);

  for(int i = 0; i < numLayers; i++) {
    neuralNetwork[i] = va_arg(args, Layer);
  }

  for(int i = 1; i < numLayers; i++) {
    computeLayer(&neuralNetwork[i - 1], &neuralNetwork[i], neuralNetwork[i].inputLayer);
  }

  va_end(args);
  return neuralNetwork;
}

int main() {

  Layer* neuralNetwork = createNeuralNetwork(
    3, 
    createLayer(1, 5, true),
    createLayer(5, 10, false),
    createLayer(10, 5, false)
  );

  printf("%f\n", neuralNetwork[2].output[0]);
}
