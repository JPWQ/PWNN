#include "main.h"

int main() {

  float input[4][2] = {{0.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 0.0f}, {1.0f, 1.0f}};
  float expectedOutput[4] = {0.0f, 1.0f, 1.0f, 0.0f};
  float** inputMatrix = malloc(sizeof(float*) * 4);
  float** outputVector = malloc(sizeof(float*) * 4);

  for(int i = 0; i < 4; i++) {
    inputMatrix[i] = malloc(sizeof(float) * 2);
    outputVector[i] = malloc(sizeof(float) * 1);
    for(int j = 0; j < 2; j++) {
      inputMatrix[i][j] = input[i][j];
    }
    for(int j = 0; j < 1; j++) {
      outputVector[i][j] = expectedOutput[i];
    }
  }


  NeuralNetwork neuralNetwork = createNeuralNetwork(
    3, 
    createLayer(4, 2),
    createLayer(2, 4),
    createLayer(4, 1)
  );

  trainNeuralNetwork(neuralNetwork, 0.1, 10000, outputVector, inputMatrix);
  float test1[2] = {0.0f, 0.0f};
  float test2[2] = {0.0f, 1.0f};
  float test3[2] = {1.0f, 0.0f};
  float test4[2] = {1.0f, 1.0f};

  testNeuralNetwork(neuralNetwork, test1);
  printf("0,0: %f\n", neuralNetwork.layers[2].output[0]);

  testNeuralNetwork(neuralNetwork, test2);
  printf("0,1: %f\n", neuralNetwork.layers[2].output[0]);

  testNeuralNetwork(neuralNetwork, test3);
  printf("1,0: %f\n", neuralNetwork.layers[2].output[0]);

  testNeuralNetwork(neuralNetwork, test4);
  printf("1,1: %f\n", neuralNetwork.layers[2].output[0]);

  // freeNN(neuralNetwork);
}
