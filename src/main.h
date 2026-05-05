#include <stdio.h>
#include <stdbool.h>
#include <stdarg.h>
#include "matrix.h"

// Each layer keeps track of how many neurons it has and an array of Neurons
typedef struct layer_t {
  float** weights;
  float* bias;
  float* output;
  int curSize;
  bool inputLayer;
} Layer;

Layer createLayer(int curSize, int nextSize, bool inputLayer);
void computeLayer(Layer* prevLayer, Layer* curLayer, bool inputLayer);
Layer* createNeuralNetwork(int numLayers, ...);
int main();
