#include <iostream>
#include <vector>
#include <numeric>

// Simple test functions to demonstrate LTO
int add_numbers(int a, int b) {
    return a + b;
}

int multiply_vector_sum(const std::vector<int>& vec, int multiplier) {
    int sum = std::accumulate(vec.begin(), vec.end(), 0);
    return sum * multiplier;
}

int main() {
    std::vector<int> numbers = {1, 2, 3, 4, 5};
    int result1 = add_numbers(10, 20);
    int result2 = multiply_vector_sum(numbers, 3);
    
    std::cout << "Addition result: " << result1 << std::endl;
    std::cout << "Vector multiplication result: " << result2 << std::endl;
    
    return 0;
}
