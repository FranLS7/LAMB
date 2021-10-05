#include <float.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <stdlib.h>
#include <time.h>
#include <chrono>
#include <string>
#include <random>
#include <algorithm>
#include "mkl.h"

#include <cube.h>
#include <common.h>

void print_menu ();

using namespace std;

int main (int argc, char **argv){
  string filename = "";
  int overhead = 2;
  int d1, d2, d3;
  int option;

  lamb::GEMM_Cube cube;

  do{
    print_menu();
    cin >> option;

    if (option == 1){
      cout << "Enter name for cube filename: ";
      cin >> filename;
    }

    else if (option == 2){
        cube.load_cube (filename, overhead);
    }

    else if (option == 3){
      cube.print_info ();
    }

    else if (option == 4){
      cout << "Insert 3 dimensions to consult: \n";
      cout << "d1: "; cin >> d1;
      cout << "d2: "; cin >> d2;
      cout << "d3: "; cin >> d3;
      // cube.load_cube (filename, overhead);
      printf("Value: %5.15f\n\n", cube.get_value (d1, d2, d3));
    }
    else
      cout << "Not a valid option, try again!" << endl;


  }
  while (option != 0);
  // cout << "Ojo que salimos!" << endl;

  return 0;
}




void print_menu (){
  cout << "==================================" << endl;
  cout << ">> Options:" << endl;
  cout << "|\t1. Specify cube_filename." << endl;
  cout << "|\t2. Load the cube." << endl;
  cout << "|\t3. Print cube information." << endl;
  cout << "|\t4. Consult one value." << endl;
  cout << "|\t0. Exit" << endl << endl;
  cout << "==================================" << endl;
  cout << "Option: ";

}