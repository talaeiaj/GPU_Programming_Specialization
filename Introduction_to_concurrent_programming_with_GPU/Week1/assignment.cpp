#include <string.h>
#include <string>
#include <iostream>
#include <fstream>
#include <istream>
#include <sstream>
#include "core_functions.hpp"
using namespace std;

const string USERNAME = "Amir";

int main(int argc, char *argv[])
{
  int results = 1;
  string username = "";

  ifstream fin(".user");
  if (!fin)
  {
    cout << "Error opening file. Shutting down..." << endl;
    return 1;
  }

  char *usernameFromFile;
  // get current line
  for (std::string line; getline(fin, line, '\n');)
  {
    usernameFromFile = &line[0];
  }

  fin.close();
  setenv("username", usernameFromFile, 1);

  if (argc > 1)
  {
    username = argv[1];
  }

  bool validUser = verifyUser(username);
  setenv("username", &USERNAME[0], 1);
  validUser = validUser && verifyUser(username);

  if (validUser)
  {
    printMessage(username);
    results = 0;
  }
  else
  {
    cout << "Error your usernames don't match check code and .user file." << endl;
  }
  return results;
}