#include <iostream>
#include <vector>

void foo(std::vector<int> v){
    std::cout << "foo size: " << v.size() << std::endl;
}

void bar(const std::vector<int>& v){
    std::cout << "bar size: " << v.size() << std::endl;
}

int main(){
    std::vector<int> my_vector = {1,2,3,4,5};
    foo(my_vector); // Pass by value (makes a copy)
    bar(my_vector); // Pass by reference (no copy)
    return 0;
}