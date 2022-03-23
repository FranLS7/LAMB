#include <float.h>
#include <stdlib.h>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>

#include "mkl.h"

#include "cube_old.h"
#include "common.h"

void print_menu ();

int main (int argc, char **argv){
  std::string filename = "";
  int overhead = 2;
  int d1, d2, d3;
  int option;
  std::string a;

  cube::GEMM_Cube cube;

  do{
    print_menu();
    std::cin >> option;

    if (option == 1){
      std::cout << "Enter name for cube filename: ";
      std::cin >> filename;
    }

    else if (option == 2){
        cube.load_cube (filename, overhead);
    }

    else if (option == 3){
      cube.print_info ();
    }

    else if (option == 4){
      std::cout << "Insert 3 dimensions to consult: \n";
      std::cout << "d1: "; std::cin >> d1;
      std::cout << "d2: "; std::cin >> d2;
      std::cout << "d3: "; std::cin >> d3;
      // cube.load_cube (filename, overhead);
      printf("Value: %5.15f\n\n", cube.get_value (d1, d2, d3));
    }
    else
      std::cout << "Not a valid option, try again!" << std::endl;


  }
  while (option != 0);
  // std::cout << "Ojo que salimos!" << endl;

  return 0;
}




void print_menu (){
  std::cout << "==================================" << std::endl;
  std::cout << ">> Options:" << std::endl;
  std::cout << "|\t1. Specify cube_filename." << std::endl;
  std::cout << "|\t2. Load the cube." << std::endl;
  std::cout << "|\t3. Print cube information." << std::endl;
  std::cout << "|\t4. Consult one value." << std::endl;
  std::cout << "|\t0. Exit" << std::endl << std::endl;
  std::cout << "==================================" << std::endl;
  std::cout << "Option: ";

}