#include "main.h"

Layer createLayer(int prevSize, int curSize, bool inputLayer) {
  // float weightsArr[curSize][prevSize];

  // The weights 2d array takes n float pointers where n is the number of neurons in the next/current layer.
  float** weightsArr = (float**)malloc(sizeof(float*) * curSize);

  // Each neruon has it's own bias so it must be a 1d array
  // taking the size of the number of neurons in the next/current layer.
  // I chose this because after forward and backward propagation each neruon
  // should be better optimized.
  // Bias array is a column vector
  float* biasArr = malloc(sizeof(float) * curSize);

  // Intialize the output array of the current layer.
  // This will be overwritten by the Compute Layer function.
  // Output array is a column vector
  float* outputArr = malloc(sizeof(float) * curSize);
  
  // For each neuron in the next/current layer initialize
  // each element of output array is a random number greater than 0 if the curLayer is not an input layer
  // each element of the weights array is a float pointer of size of the previous layer
  // each element of the biass array is a random number greater than 0

  for(int i = 0; i < curSize; i++) {
    // If the current layer is the input layer assign values to it's output array.
    // This is neccessary because when computeLayer() is called in createNeuralNetwork()
    // it skips the first array because there is no previous layer for the input layer.
    if(inputLayer) {
      outputArr[i] = ((float)rand() / RAND_MAX);
    }
    weightsArr[i] = (float*)malloc(sizeof(float) * prevSize);

    biasArr[i] = (((float)rand() / RAND_MAX) * 50.0f - 25.0f);

    // For each neuron in the previous layer initialize
    // the ith pointer to a random number greater than 0
    for(int j = 0; j < prevSize; j++) {
      weightsArr[i][j] = (((float)rand() / RAND_MAX) * 50.0f - 25.0f);
    }
  }

  Layer newLayer = {weightsArr, biasArr, outputArr, curSize};
  return newLayer;
}

// Using the weights and bias from the previous layer compute the output of the current layer
void computeLayer(Layer* prevLayer, Layer* curLayer) {
  // Using the formula Z = W*A + B

  // Taking the transpose is unnecessary because
  // the weights matrix uses the row vector convention

  // Then using the sigmoid activation function apply it to each output in the output vector
  float* result = matrixMult(prevLayer->weights, prevLayer->output, prevLayer->curSize, curLayer->curSize);
  for(int i = 0; i < prevLayer->curSize; i++) {
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
    computeLayer(&neuralNetwork[i - 1], &neuralNetwork[i]);
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

    for(int i = 0; i < 5; i++) {
      printf("%f\n", neuralNetwork[2].output[i]);
    }
}
