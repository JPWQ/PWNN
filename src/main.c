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
  float* outputArr = malloc(sizeof(float) * curSize);
  
  // For each neuron in the next/current layer initialize
  // each element of output array is a random number greater than 0 if the curLayer is not an input layer
  // each element of the weights array is a float pointer of size of the previous layer
  // each element of the biass array is a random number greater than 0

  // For each neuron in the previous layer initialize
  // the ith pointer to a random number greater than 0
  for(int i = 0; i < curSize; i++) {
    if(!inputLayer) {
      outputArr[i] = (((float)rand() / RAND_MAX) * 50.0f - 25.0f);
    }

    weightsArr[i] = (float*)malloc(sizeof(float) * prevSize);

    biasArr[i] = (((float)rand() / RAND_MAX) * 50.0f - 25.0f);

    for(int j = 0; j < prevSize; j++) {
      weightsArr[i][j] = (((float)rand() / RAND_MAX) * 50.0f - 25.0f);
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
      curLayer->output[i] = (((float)rand() / RAND_MAX));
    }
    return;
  }

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
  Layer** neuralNetwork = malloc(sizeof(Layer*));
  neuralNetwork[0] = malloc(sizeof(Layer) * numLayers);

  va_list args;
  va_start(args, numLayers);

  for(int i = 0; i < numLayers; i++) {
    neuralNetwork[0][i] = va_arg(args, Layer);
  }

  for(int i = 1; i < numLayers; i++) {
    computeLayer(&neuralNetwork[0][i - 1], &neuralNetwork[0][i], neuralNetwork[0][i].inputLayer);
  }

  va_end(args);
  return *neuralNetwork;
}

int main() {

  // Layer* neuralNetwork = createNeuralNetwork(
  //   3, 
  //   createLayer(1, 5, true),
  //   createLayer(5, 10, false),
  //   createLayer(10, 5, false)
  // );

    for(int i = 0; i < 5; i++) {
      // printf("%f\n", neuralNetwork[2].output[i]);
    }
  float testMatrix[2][3] = {{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}};
  float testVector[3] = {7.0f, 8.0f, 9.0f};
  float* testOutput;

  float** matrix = malloc(sizeof(float*) * 2);
  float* vector = malloc(sizeof(float) * 3);
  for(int i = 0; i < 2; i++) {
    matrix[i] = malloc(sizeof(float) * 3);
    for(int j = 0; j < 3; j++) {
      matrix[i][j] = testMatrix[i][j];
      vector[j] = testVector[j];
    }
  }

  // printf("%f", testMatrix[1][1]);
  testOutput = matrixMult(matrix, vector, 2, 3);
  for(int i = 0; i < 3; i++) {
    printf("%f\n", testOutput[i]);
    printf("Test: %f\n", vector[i]);
  }
}
