#include <iostream>
#include <vector>
#include <cassert>
#include "relay_bp_decoder.h"

int main() {
    std::cout << "Testing C++ Relay-BP Implementation..." << std::endl;

    // Repetition code d=3
    // H = [[1, 1, 0], [0, 1, 1]]
    int n_checks = 2;
    int n_vars = 3;
    
    // CSR Format
    std::vector<int> indptr = {0, 2, 4};
    std::vector<int> indices = {0, 1, 1, 2};
    // Data is implied all 1s for binary matrix
    
    double p = 0.1;
    std::vector<double> priors = {p, p, p};
    
    RelayDecoder decoder(indptr, indices, n_checks, n_vars, priors, 10, 5, 10);
    
    // Test 1: Error on bit 0 -> Syndrome [1, 0]
    std::vector<int> syndrome = {1, 0};
    std::vector<int> result;
    
    bool success = decoder.decode(syndrome, result);
    
    std::cout << "Syndrome [1, 0] Success: " << success << std::endl;
    std::cout << "Result: ";
    for (int x : result) std::cout << x << " ";
    std::cout << std::endl;
    
    if (success && result[0] == 1 && result[1] == 0 && result[2] == 0) {
        std::cout << "PASS" << std::endl;
    } else {
        std::cout << "FAIL" << std::endl;
    }

    return 0;
}
