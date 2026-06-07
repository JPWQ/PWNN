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
    // input layer prev size is the number of samples
    createLayer(4, 2, "sigmoid"),
    createLayer(2, 4, "sigmoid"),
    // ouput layer cur size is 1 becuase output is binary
    createLayer(4, 1, "sigmoid")
  );

  trainNeuralNetwork(neuralNetwork, 0.1, 1000000, outputVector, inputMatrix);
  float test1[2] = {0.0f, 0.0f};
  float test2[2] = {0.0f, 1.0f};
  float test3[2] = {1.0f, 0.0f};
  float test4[2] = {1.0f, 1.0f};

  testNeuralNetwork(neuralNetwork, test1, *outputVector[0]);

  testNeuralNetwork(neuralNetwork, test2, *outputVector[1]);

  testNeuralNetwork(neuralNetwork, test3, *outputVector[2]);

  testNeuralNetwork(neuralNetwork, test4, *outputVector[3]);

  freeNN(neuralNetwork);
  printf("\n");

  float** MNISTInputMatrix = malloc(sizeof(float*) * 60000);
  float** MNISTLabels = malloc(sizeof(float*) * 60000);

  // FILE* train = fopen("~/Downloads/mnist_train.csv", "r");
  // char line[8192];
  // fgets(line, sizeof(line), train);
  // int count = 0;
  // while (fgets(line, sizeof(line), train)) {
  //   char* token = strtok(line, ",");
  //   int label = atoi(token);
  //   
  //   MNISTInputMatrix[count] = malloc(sizeof(float) * 784);
  //   MNISTLabels[count] = malloc(sizeof(float) * 10);
  //
  //   for (int i = 0; i < 784; i++) {
  //     token = strtok(NULL, ",");
  //     MNISTInputMatrix[count][i] = atof(token) / 255.0f;
  //   }
  //
  //   for (int i = 0; i < 10; i++) {
  //     MNISTLabels[count][i] = 0.0f;
  //   }
  //   MNISTLabels[count][label] = 1.0f;
  //
  //   count++;
  // }
  // fclose(train);

  // NeuralNetwork neuralNetwork1 = createNeuralNetwork(
  //   3,
  //   // input layer prev size is the number of samples
  //   createLayer(60000, 784, "relu"),
  //   createLayer(784, 128, "relu"),
  //   // ouput layer cur size is 1 becuase output is binary
  //   createLayer(128, 10, "sigmoid")
  // );
  //
  // trainNeuralNetwork(neuralNetwork1, 0.1, 100000, MNISTLabels, MNISTInputMatrix);
  //
}


