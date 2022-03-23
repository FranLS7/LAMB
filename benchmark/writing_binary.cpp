#include <iostream>
#include <fstream>

using namespace std;

struct Student {
  int roll_no;
  double grade;
};

int main() {
  ofstream wf("student.dat", ios::out | ios::binary);
  if (!wf) {
    cout << "Cannot open file!" << endl;
    return -1;
  }

  Student wstu[3];
  wstu[0].roll_no = 1;
  wstu[0].grade = 0.8;

  wstu[1].roll_no = 2;
  wstu[1].grade = 4.5;

  wstu[2].roll_no = 3;
  wstu[2].grade = 4.5;

  for (int i = 0; i < 3; i++) 
    wf.write ((char *) &wstu[i], sizeof(Student));

  wf.close();
  if (!wf.good()) {
    cout << "Error while writing!" << endl;
    return -1;
  }

  ifstream rf("student.dat", ios::in | ios::binary);
  if (!rf) {
    cout << "Cannot open file for reading!" << endl;
    return -1;
  }

  Student rstu[3];
  // rf.read((char*)&rstu[0], sizeof(Student));
  for(int i = 0; i < 3; i++)
    rf.read((char*) &rstu[i], sizeof(Student));
  
  rf.close();
  if (!rf.good()) {
    cout << "Error occurred at reading time!" << endl;
    return -1;
  }

  cout<<"Student's Details:" << endl;
  for(int i=0; i < 3; i++) {
    cout << "Roll No: " << rstu[i].roll_no << endl;
    cout << "Grade: " << rstu[i].grade << endl;
    cout << endl;
  }
   
  return 0;
}