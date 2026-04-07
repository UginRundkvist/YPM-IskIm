#include <iostream>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <fstream>
#include <vector>
#include <chrono>

using namespace std;
using namespace chrono;

// ==================== КОНСТАНТЫ ====================
const int BASE_SIZE = 5000;           // Размер базового вектора (пункт 4)
const int MIN_VALUE = 0;              // Минимальное значение случайного числа
const int MAX_VALUE = 5000;           // Максимальное значение случайного числа
const int REPEAT_COUNT = 100;         // Количество повторов для малых n (пункт 8)

// Значения параметра n (пункт 5)
const int N_VALUES[] = { 100, 200, 300, 400, 500, 750, 1000, 2000, 3000, 4000, 5000, 10000, 20000 };
const int N_COUNT = sizeof(N_VALUES) / sizeof(N_VALUES[0]);

// ==================== АЛГОРИТМЫ СОРТИРОВКИ ====================

// 1. Сортировка вставками (алгоритм 1)
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

// Восстановление свойства кучи (для пирамидальной сортировки)
void restore(int arr[], int n, int i) {
    int largest = i;
    int left = 2 * i + 1;
    int right = 2 * i + 2;

    if (left < n && arr[left] > arr[largest])
        largest = left;
    if (right < n && arr[right] > arr[largest])
        largest = right;

    if (largest != i) {
        swap(arr[i], arr[largest]);
        restore(arr, n, largest);
    }
}

// Построение кучи
void buildHeap(int arr[], int n) {
    for (int i = n / 2 - 1; i >= 0; i--)
        restore(arr, n, i);
}

// 5. Пирамидальная сортировка (алгоритм 5)
void heapSort(int arr[], int n) {
    buildHeap(arr, n);
    for (int i = n - 1; i > 0; i--) {
        swap(arr[0], arr[i]);
        restore(arr, i, 0);
    }
}

// ==================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ====================

// Генерация случайного массива заданного размера
void generateArray(int* arr, int size) {
    for (int i = 0; i < size; i++) {
        arr[i] = rand() % (MAX_VALUE - MIN_VALUE + 1) + MIN_VALUE;
    }
}

// Копирование части массива (для взятия "головы" вектора)
void copyArray(int* dest, int* src, int n) {
    for (int i = 0; i < n; i++) {
        dest[i] = src[i];
    }
}

// Замер времени сортировки с возможными повторами (пункт 8)
double measureTime(void (*sortFunc)(int[], int), int* arr, int n, int repeats) {
    auto start = high_resolution_clock::now();
    
    for (int r = 0; r < repeats; r++) {
        // Делаем копию массива, чтобы не сортировать уже отсортированный
        int* tempArr = new int[n];
        copyArray(tempArr, arr, n);
        sortFunc(tempArr, n);
        delete[] tempArr;
    }
    
    auto end = high_resolution_clock::now();
    duration<double> elapsed = end - start;
    return elapsed.count() / repeats;  // Среднее время за один проход
}

// ==================== РАБОТА С ФАЙЛАМИ ====================

// Сохранение результатов в файл (пункт 6)
void saveResultsToFile(double results[][3], const char* filename) {
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "Ошибка: не удалось открыть файл для записи!" << endl;
        return;
    }
    
    file << "n\tInsertionSort\tHeapSort\n";
    for (int i = 0; i < N_COUNT; i++) {
        file << results[i][0] << "\t" 
             << fixed << setprecision(6) << results[i][1] << "\t"
             << results[i][2] << "\n";
    }
    
    file.close();
    cout << "\nРезультаты сохранены в файл: " << filename << endl;
}

// Загрузка результатов из файла (пункт 6)
void loadResultsFromFile(const char* filename) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Ошибка: не удалось открыть файл для чтения!" << endl;
        return;
    }
    
    string line;
    cout << "\n=== Результаты из файла " << filename << " ===\n";
    while (getline(file, line)) {
        cout << line << endl;
    }
    
    file.close();
}

// Вывод таблицы на экран (пункт 7)
void printTable(double results[][3]) {
    cout << "\n" << string(70, '=') << endl;
    cout << " " << setw(10) << "n"
         << " | " << setw(20) << "Вставками (сек)"
         << " | " << setw(20) << "Пирамид. (сек)" << endl;
    cout << string(70, '-') << endl;
    
    for (int i = 0; i < N_COUNT; i++) {
        cout << " " << setw(10) << (int)results[i][0]
             << " | " << setw(20) << fixed << setprecision(8) << results[i][1]
             << " | " << setw(20) << results[i][2] << endl;
    }
    cout << string(70, '=') << endl;
}

// ==================== ОСНОВНАЯ ФУНКЦИЯ ====================

int main() {
    srand(static_cast<unsigned int>(time(nullptr)));
    
    // Массив для хранения результатов [n][время1][время2]
    double results[N_COUNT][3];
    
    // 1. Генерируем базовый вектор длины BASE_SIZE = 5000 (пункт 4)
    int* baseArray = new int[BASE_SIZE];
    generateArray(baseArray, BASE_SIZE);
    cout << "Сгенерирован базовый вектор из " << BASE_SIZE << " случайных чисел\n";
    cout << "Диапазон значений: [" << MIN_VALUE << ", " << MAX_VALUE << "]\n";
    
    // 2. Для каждого значения n (пункт 5)
    for (int idx = 0; idx < N_COUNT; idx++) {
        int n = N_VALUES[idx];
        cout << "\nОбработка n = " << n << "...";
        
        // Определяем, сколько повторов делать (пункт 8)
        int repeats = REPEAT_COUNT;
        if (n >= 10000) repeats = 1;      // Большие n — без повторов
        else if (n >= 5000) repeats = 5;  // Средние n — небольшие повторы
        else if (n >= 1000) repeats = 20; // Малые n — больше повторов
        else repeats = 100;               // Очень малые n — много повторов
        
        // Создаём массив для текущего теста
        int* testArray = new int[n];
        
        // Берём "голову" базового вектора (если n <= BASE_SIZE)
        if (n <= BASE_SIZE) {
            copyArray(testArray, baseArray, n);
        } 
        else {
            // Для n > 5000 генерируем новый массив (пункт 5)
            generateArray(testArray, n);
        }
        
        // Замеряем время для сортировки вставками
        double timeInsertion = measureTime(insertionSort, testArray, n, repeats);
        
        // Замеряем время для пирамидальной сортировки
        double timeHeap = measureTime(heapSort, testArray, n, repeats);
        
        // Сохраняем результаты
        results[idx][0] = n;
        results[idx][1] = timeInsertion;
        results[idx][2] = timeHeap;
        
        cout << " готово. (повторов: " << repeats << ")";
        cout << " Вставки: " << fixed << setprecision(6) << timeInsertion << "с";
        cout << " Пирамида: " << timeHeap << "с";
        
        delete[] testArray;
    }
    
    // 3. Выводим таблицу результатов (пункт 7)
    printTable(results);
    
    // 4. Сохраняем в файл (пункт 6)
    saveResultsToFile(results, "sort_results.txt");
    
    // 5. Демонстрация чтения из файла (пункт 6)
    char choice;
    cout << "\nПоказать содержимое файла? (y/n): ";
    cin >> choice;
    if (choice == 'y' || choice == 'Y') {
        loadResultsFromFile("sort_results.txt");
    }
    
    // 6. Анализ эффективности (пункт 11)
    cout << "\n=== АНАЛИЗ ЭФФЕКТИВНОСТИ ===\n";
    int fasterCount = 0;
    for (int i = 0; i < N_COUNT; i++) {
        if (results[i][2] < results[i][1]) {
            fasterCount++;
        }
    }
    cout << "Пирамидальная сортировка оказалась быстрее в " 
         << fasterCount << " из " << N_COUNT << " случаев\n";
    
    if (fasterCount > N_COUNT / 2) {
        cout << "Вывод: Пирамидальная сортировка (O(n log n)) предпочтительнее для больших массивов\n";
    } else {
        cout << "При малых n сортировка вставками может быть эффективнее\n";
    }

    delete[] baseArray;
    
    cout << "\nПрограмма завершена. Результаты сохранены в sort_results.txt\n";
    
    return 0;
}