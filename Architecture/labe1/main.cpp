#include <iostream>
#include <random>
#include <ctime>

int main() {
    const int SIZE = 20000;
    const int MIN_VALUE = 0;
    const int MAX_VALUE = 5000;
    
   
    int* arr = new int[SIZE];
    
    //Инициализируем генератор случайных чисел
    std::random_device rd;
    std::mt19937 gen(rd());  
    std::uniform_int_distribution<> dis(MIN_VALUE, MAX_VALUE);
    
    for (int i = 0; i < SIZE; i++) {
        arr[i] = dis(gen);
    }
    
    // Выводим первые 20 элементов для проверки
    std::cout << "Первые 20 элементов массива:" << std::endl;
    for (int i = 0; i < 20 && i < SIZE; i++) {
        std::cout << arr[i] << " ";
    }
    std::cout << std::endl;
    
    // Выводим последние 20 элементов для проверки
    std::cout << "Последние 20 элементов массива:" << std::endl;
    for (int i = SIZE - 20; i < SIZE; i++) {
        std::cout << arr[i] << " ";
    }
    std::cout << std::endl;
    
    // Освобождаем память
    delete[] arr;
    
    return 0;
}