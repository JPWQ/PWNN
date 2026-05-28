#include "utils.h"

bool stringCmp(char* string1, char* string2) {
  if (strlen(string1) != strlen(string2)) {
    return false;
  }
  else {
    for (int i = 0; i < (int)strlen(string1); i++) {
      if (string1[i] != string2[i]) {
        return false;
      }
    }
    return true;
  }
}
