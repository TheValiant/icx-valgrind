#include <iostream>

// Another translation unit for LTO testing
extern int add_numbers(int a, int b);

int helper_function(int x) {
    return add_numbers(x, x * 2);
}

void print_helper_result(int input) {
    int result = helper_function(input);
    std::cout << "Helper result for " << input << ": " << result << std::endl;
}
