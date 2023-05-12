//
// Created by Chancellor Pascale on 1/31/21.
//
#include<mutex>
#include<atomic>
#include "assignment.h"


std::mutex mu;

void executeTicketingSystemParticipation() {
	// For debugging purposes you might want to output the thread's ticketNumber

	int i = currentTicketNumber++;
	int ticketNumber = i;
	
	std::cout << "Current thread ticket number: " << ticketNumber << "\n";
	
	std::string outputFileName = "output-" + currentPartId + ".txt";
	// NOTE: Do not remove this output to file statement as it is used to grade assignment,
	// so it should be called by each thread

	mu.lock();

	std::ofstream outputFile;
	outputFile.open (outputFileName, std::ofstream::app);
	outputFile << "C++11: Thread retrieved ticket number: " << ticketNumber << " started.\n";
	
	// wait until your ticket number has been called output your ticket number and the current time

	// NOTE: Do not remove this output to file statement as it is used to grade assignment,
	// so it should be called by each thread
	outputFile << "C++11: Thread with ticket number: " << ticketNumber << " completed.\n";
	outputFile.close();

	mu.unlock();

	// Increment currentTicketNumber variable to allow other threads to do their job
	currentTicketNumber++;
}

int runSimulation() {
	int result = 0;
	std::string userFromFile = getUsernameFromUserFile();
	if(USERNAME == currentUser && USERNAME == userFromFile){
		std::cout << "Simple user verification completed successfully.\n";
		std::atomic<int> currentTicketNumber;
		//ticketMachineNumber = -1;
		currentTicketNumber = -1;
		
		std::thread threads[currentNumThreads];
		
		for(int threadIndex = 0; threadIndex < currentNumThreads; ++threadIndex){
			// This is where you will start threads that will participate in a ticketing system
			// have the thread run the executeTicketingSystemParticipation function

			threads[threadIndex] = std::thread(executeTicketingSystemParticipation);
		}
		
		// The code will also need to know when all threads have completed their work
		std::this_thread::sleep_for(std::chrono::seconds(1));
		manageTicketingSystem();

	} else {
		std::cout << "Simple user verification completed failed, code will not be executed.\n";
	}
	return result;
}

// Utility function for retrieving a user name from the .user file
std::string getUsernameFromUserFile(){
	std::string line;
	std::ifstream userFile (".user");
	if (userFile.is_open())
	{
		std::getline (userFile,line);
		userFile.close();
	}
	std::cout << "user from .user file: " << line << "\n";
	return line;
}

// Centralized logic for managing the ticketing machine that is the basis for threads executing work in the order that
//	they "pulled" the ticket from the machine
int manageTicketingSystem() {
	std::string outputFileName = "output-" + currentPartId + ".txt";
	std::ofstream outputFile;
	outputFile.open (outputFileName, std::ofstream::app);
	outputFile << "C++11: Signaling threads to do work.\n";
	
	// Increment a ticket number shared by a number of threads and check that no active threads are running

	currentTicketNumber++;

	// Wait for all threads to complete
	
	outputFile << "C++11: All threads completed.\n";
	outputFile.close();
	return 0;
}

int main(int argc, char *argv[]) {
	int numThreads = 1;
	std::string user = "Amir";
	std::string partId = "test";
	std::cout << "Starting assignment main function\n";
	
	if (argc > 3) {
		std::cout << "Parsing command line arguments\n";
		numThreads = atoi(argv[1]);
		user = argv[2];
		partId = argv[3];
	}
	
	currentNumThreads = numThreads;
	currentUser = user;
	currentPartId = partId;
	
	// Note this implementation is based on a single code file and use of static and shared variables,
	// other implementations could be designed and are encouraged
	
	runSimulation();
}
