#include <iostream>
#include <cstdlib>
#include <ctime>
#include <iomanip>

void insertionSort(int arr[], int n) {
    for (int i = 1; i < n; i++) {
        int key = arr[i];  
        int j = i - 1;
        
        while (j >= 0 && arr[j] > key) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = key;
    }
}

void restore(int arr[], int n, int i) {
    int largest = i;        
    int left = 2 * i + 1;   
    int right = 2 * i + 2;  
    
    if (left < n && arr[left] > arr[largest]) {
        largest = left;
    }
    
    if (right < n && arr[right] > arr[largest]) {
        largest = right;
    }
    
    if (largest != i) {
        std::swap(arr[i], arr[largest]);
        restore(arr, n, largest);
    }
}

void buildHeap(int arr[], int n) {
    for (int i = n / 2 - 1; i >= 0; i--) {
        restore(arr, n, i);
    }
}

void heapSort(int arr[], int n) {
    buildHeap(arr, n);
    
    for (int i = n - 1; i > 0; i--) {
        std::swap(arr[0], arr[i]);
        restore(arr, i, 0);
    }
}

int main() {
    const int SIZE = 20000;
    const int MIN_VALUE = 0;
    const int MAX_VALUE = 5000;
    
    srand(static_cast<unsigned int>(time(nullptr)));
    
    int* arr1 = new int[SIZE];  
    int* arr2 = new int[SIZE]; 
    
    for (int i = 0; i < SIZE; i++) {
        int value = rand() % (MAX_VALUE - MIN_VALUE + 1) + MIN_VALUE;
        arr1[i] = value;
        arr2[i] = value;
    }
    
    std::cout << "Сравнение сортировок" << std::endl;    
    std::cout << "\nПервые 20 элементов ДО сортировки:" << std::endl;
    for (int i = 0; i < 20; i++) {
        std::cout << std::setw(5) << arr1[i] << " ";
        if ((i + 1) % 10 == 0) {
            std::cout << std::endl;
        }
    }
    std::cout << std::endl;
    
    std::cout << "\nСортировка вставками" << std::endl;
    
    clock_t start1 = clock();
    insertionSort(arr1, SIZE);
    clock_t end1 = clock();
    double time1 = static_cast<double>(end1 - start1) / CLOCKS_PER_SEC;
    
    std::cout << "Первые 20 элементов после сортировки вставками:" << std::endl;
    for (int i = 0; i < 20; i++) {
        std::cout << std::setw(5) << arr1[i] << " ";
        if ((i + 1) % 10 == 0) {
            std::cout << std::endl;
        }
    }
    std::cout << "\nВремя выполнения: " << time1 << " секунд" << std::endl;
    std::cout << "Сложность алгоритма: O(n²) в худшем случае" << std::endl;
    
    std::cout << "\nПирамидальная сортировка" << std::endl;
    
    clock_t start2 = clock();
    heapSort(arr2, SIZE);
    clock_t end2 = clock();
    double time2 = static_cast<double>(end2 - start2) / CLOCKS_PER_SEC;
    
    std::cout << "Первые 20 элементов после пирамидальной сортировки:" << std::endl;
    for (int i = 0; i < 20; i++) {
        std::cout << std::setw(5) << arr2[i] << " ";
        if ((i + 1) % 10 == 0) {
            std::cout << std::endl;
        }
    }
    std::cout << "\nВремя выполнения: " << time2 << " секунд" << std::endl;
    std::cout << "Сложность алгоритма: O(n log n) в любом случае" << std::endl;
    
    std::cout << "\nСравнение эффективности" << std::endl;
    std::cout << "Сортировка вставками:    " << time1 << " сек" << std::endl;
    std::cout << "Пирамидальная сортировка: " << time2 << " сек" << std::endl;
    
    if (time1 < time2) {
        std::cout << "Сортировка вставками быстрее на " 
                  << ((time2 - time1) / time2 * 100) << "%" << std::endl;
    } else {
        std::cout << "Пирамидальная сортировка быстрее на " 
                  << ((time1 - time2) / time1 * 100) << "%" << std::endl;
    }
    
    delete[] arr1;
    delete[] arr2;
    
    return 0;
}