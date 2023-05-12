//
// Created by Chancellor Pascale on 1/31/21.
//
#include <thread>
#include <iostream>
#include <fstream>
#include <string>

#ifndef CPP_INITIAL_CODE_ASSIGNMENT_H
#define CPP_INITIAL_CODE_ASSIGNMENT_H

const std::string USERNAME = "Amir";
static int currentTicketNumber; // This could be an atomic variable
// Based on the implementation, the student may want to create a shared variable for managing the next ticket number
// to be assigned to a thread
static std::string currentPartId;
static std::string currentUser;
static int currentNumThreads;

void executeTicketingSystemParticipation();
int runSimulation();
std::string getUsernameFromUserFile();
int manageTicketingSystem();

#endif //CPP_INITIAL_CODE_ASSIGNMENT_H
