#include <stdlib.h>
#include <string>
#include <string.h>
#include <iostream>
using namespace std;

void printMessage(string text)
{
  cout << "Hello " + text + "!" << endl;
}

bool verifyUser(std::string username)
{
  string usernameEnvVar = getenv("username");
  return username.compare(usernameEnvVar) == 0;
}