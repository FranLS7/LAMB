#include <iostream>
#include <unordered_map>
#include <vector>
#include <string>

struct vectorHasher {
  int operator() (const std::vector<int> &V) const {
    int hash = V.size();
    for (auto &i : V) {
      hash ^= i + 0x9e3779b9 + (hash << 6) + (hash >> 2);
      std::cout << "hash: " << hash << std::endl;
    }
    return hash;
  }
};

std::string serialiseVector (std::vector<int> v){
  std::string buffer;

  for (auto const &x : v){
    buffer.append (std::to_string(x) + "/");
  }

  return buffer;
}

int main() {
  std::unordered_map<std::string, std::vector<int>> hashMap;

  std::vector<int> dims {100, 200, 300, 400, 500};
  std::vector<int> dims2 {50, 200, 400, 700, 200};
  std::vector<int> dims3 {30, 90, 700, 3000, 205};

  std::string buffer = serialiseVector (dims);
  hashMap [buffer] = dims;

  std::cout << ">> " << buffer << std::endl;
  std::cout << ">> size: " << buffer.size() << std::endl;

  buffer = serialiseVector (dims2);
  hashMap [buffer] = dims2;

  buffer = serialiseVector (dims3);
  hashMap [buffer] = dims3;

  for (auto const &i : hashMap) {
    std::cout << i.first << std::endl;
  }

  return 0;
}