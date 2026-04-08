using System;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Windows.Forms;
using System.Xml.Linq;

namespace SortingLab
{
    public partial class Form1 : Form
    {
        // Константы
        private const int BASE_SIZE = 20000;
        private const int MIN_VALUE = 0;
        private const int MAX_VALUE = 5000;
        private readonly int[] N_VALUES = { 100, 200, 300, 400, 500, 750, 1000, 2000, 3000, 4000, 5000, 10000, 20000 };

        private int[] baseArray;
        private double[,] results;
        private Random rand = new Random();

        public Form1()
        {
            InitializeComponent();

            // Настройка таблицы
            dgvResults.ColumnCount = 3;
            dgvResults.Columns[0].Name = "n";
            dgvResults.Columns[0].HeaderText = "Длина вектора (n)";
            dgvResults.Columns[0].Width = 120;
            dgvResults.Columns[1].Name = "InsertionTime";
            dgvResults.Columns[1].HeaderText = "Сортировка вставками (сек)";
            dgvResults.Columns[1].Width = 180;
            dgvResults.Columns[2].Name = "HeapTime";
            dgvResults.Columns[2].HeaderText = "Пирамидальная сортировка (сек)";
            dgvResults.Columns[2].Width = 180;

            // Подпись на кнопки
            btnGenerate.Click += BtnGenerate_Click;
            btnRun.Click += BtnRun_Click;
            btnSave.Click += BtnSave_Click;
            btnLoad.Click += BtnLoad_Click;

            btnRun.Enabled = false;
            results = new double[N_VALUES.Length, 3];

            Log("Программа запущена. Нажмите 'Сгенерировать'.");
        }

        // АЛГОРИТМЫ СОРТИРОВКИ

        // сортировка вставками
        private void InsertionSort(int[] unsortedArray)
        {
            for (int i = 1; i < unsortedArray.Length; i++)
            {
                int key = unsortedArray[i];
                int j = i - 1;
                while (j >= 0 && unsortedArray[j] > key)
                {
                    unsortedArray[j + 1] = unsortedArray[j];
                    j--;
                }
                unsortedArray[j + 1] = key;
            }
        }

        // функция меняет местами эллементы в массиве(в данном случае в дереве)
        private void Swap(ref int a, ref int b)
        {
            int temp = a;
            a = b;
            b = temp;
        }

        //Restore восстанавливает свойство дерева для узла i, предполагая, что его левое и правое поддеревья уже являются корректными кучами.
        private void Restore(int[] unsortedArray, int elementsInHeap, int initiaIndex)
        {
            int largestElemIndex = initiaIndex;
            int leftChildIndex = 2 * initiaIndex + 1;
            int rightChildIndex = 2 * initiaIndex + 2;

            if (leftChildIndex < elementsInHeap && unsortedArray[leftChildIndex] > unsortedArray[largestElemIndex])
                largestElemIndex = leftChildIndex;

            if (rightChildIndex < elementsInHeap && unsortedArray[rightChildIndex] > unsortedArray[largestElemIndex])
                largestElemIndex = rightChildIndex;

            if (largestElemIndex != initiaIndex)
            {
                Swap(ref unsortedArray[initiaIndex], ref unsortedArray[largestElemIndex]);
                Restore(unsortedArray, elementsInHeap, largestElemIndex);
            }
        }

        // Функия сортировки кучами
        private void HeapSort(int[] unsortedArray)
        {
            int arrLength = unsortedArray.Length;

            for (int initiaIndex = arrLength / 2 - 1; initiaIndex >= 0; initiaIndex--)
                Restore(unsortedArray, arrLength, initiaIndex);

            for (int i = arrLength - 1; i > 0; i--)
            {
                Swap(ref unsortedArray[0], ref unsortedArray[i]);
                Restore(unsortedArray, i, 0);
            }
        }

        //  ВСПОМОГАТЕЛЬНЫЕ

        // генерация массива случайных чисел от 0 до 5000
        private int[] GenerateRandomArray(int arrSize)
        {
            int[] unsortedArray = new int[arrSize];
            for (int i = 0; i < arrSize; i++) unsortedArray[i] = rand.Next(MIN_VALUE, MAX_VALUE + 1);
            return unsortedArray;
        }

        // Функция для создания меньшиж по длинне неотсортированных массивов ((переименовать)
        private int[] CopyArray(int[] sourceArray, int arrLength)
        {
            int[] childArray = new int[arrLength];
            Array.Copy(sourceArray, childArray, arrLength);
            return childArray;
        }

        // Измерение времени сортировок
        private double MeasureTime(Action<int[]> sortFunc, int[] childArray, int repeats)
        {
            Stopwatch sw = Stopwatch.StartNew();
            for (int r = 0; r < repeats; r++)
            {
                int[] temp = CopyArray(childArray, childArray.Length);
                sortFunc(temp);
            }
            sw.Stop();
            return sw.Elapsed.TotalSeconds / repeats;
        }

        // Функция для определения кол-ва повторений для массивов с разной длинной
        private int GetRepeats(int n)
        {
            if (n >= 10000) return 1;
            if (n >= 3000) return 5;
            if (n >= 1000) return 20;
            return 100;
        }

        // Функция для логов
        private void Log(string msg)
        {
            txtLog.AppendText($"[{DateTime.Now:HH:mm:ss}] {msg}\n");
            txtLog.ScrollToCaret();
        }

        // обновление прогресс бара
        private void UpdateProgress(int current, int total)
        {
            progressBar.Maximum = total;
            progressBar.Value = current;
            lblStatus.Text = $"Готово: {current} из {total}";
        }

        //ОБРАБОТЧИКИ КНОПОК

        // кнопка генерации массива
        private void BtnGenerate_Click(object sender, EventArgs e)
        {
            Log("Генерация базового массива...");
            baseArray = GenerateRandomArray(BASE_SIZE);
            Log($"Сгенерирован массив из {BASE_SIZE} чисел (0-{MAX_VALUE})");
            btnRun.Enabled = true;
            dgvResults.Rows.Clear();
            MessageBox.Show($"Массив из {BASE_SIZE} элементов готов!", "Готово");
        }

        // кнопка для сортировки и оценки времени
        private void BtnRun_Click(object sender, EventArgs e)
        {
            if (baseArray == null)
            {
                MessageBox.Show("Сначала сгенерируйте массив!");
                return;
            }

            btnRun.Enabled = false;
            btnGenerate.Enabled = false;
            btnSave.Enabled = false;
            dgvResults.Rows.Clear();


            Log("НАЧАЛО ТЕСТИРОВАНИЯ");
            for (int lengthIdx = 0; lengthIdx < N_VALUES.Length; lengthIdx++)
            {
                int currentArrayLength = N_VALUES[lengthIdx];
                int repeats = GetRepeats(currentArrayLength);

                UpdateProgress(lengthIdx + 1, N_VALUES.Length);
                Log($"Тест n={currentArrayLength} (повторов: {repeats})...");

                Application.DoEvents();

                int[] testArray = CopyArray(baseArray, currentArrayLength);
                double timeIns = MeasureTime(InsertionSort, testArray, repeats);
                double timeHeap = MeasureTime(HeapSort, testArray, repeats);

                results[lengthIdx, 0] = currentArrayLength;
                results[lengthIdx, 1] = timeIns;
                results[lengthIdx, 2] = timeHeap;

                // вывод в таблицу
                dgvResults.Rows.Add(currentArrayLength, timeIns.ToString("F8"), timeHeap.ToString("F8"));
                Log($"  Вставки: {timeIns:F6}с | Пирамида: {timeHeap:F6}с");
            }

            UpdateProgress(0, 0);
            lblStatus.Text = "Готово!";
            btnRun.Enabled = true;
            btnGenerate.Enabled = true;
            btnSave.Enabled = true;
            Log("ТЕСТИРОВАНИЕ ЗАВЕРШЕНО");

            int fasterCount = 0;
            for (int i = 0; i < N_VALUES.Length; i++)
                if (results[i, 2] < results[i, 1]) fasterCount++;

            Log($"Пирамидальная сортировка быстрее в {fasterCount} из {N_VALUES.Length} случаев");
        }

        // кнопка для сохранения результатов в файл
        private void BtnSave_Click(object sender, EventArgs e)
        {
            if (dgvResults.Rows.Count == 0)
            {
                MessageBox.Show("Нет результатов для сохранения!", "Ошибка",
                    MessageBoxButtons.OK, MessageBoxIcon.Warning);
                return;
            }

            //сохраняем файл
            SaveFileDialog dlg = new SaveFileDialog();
            dlg.Filter = "CSV файлы (*.csv)|*.csv";
            dlg.FileName = $"sort_results_{DateTime.Now:yyyyMMdd_HHmmss}.csv";

            // ждем пока пользователь сохранит файл
            if (dlg.ShowDialog() == DialogResult.OK)
            {
                try
                {
                    using (StreamWriter sw = new StreamWriter(dlg.FileName))
                    {
                        // Заголовки
                        sw.WriteLine("n;InsertionSort_sec;HeapSort_sec");

                        for (int i = 0; i < N_VALUES.Length; i++)
                        {
                            // разделитель точка
                            string insertionSortTime = results[i, 1].ToString("F8").Replace('.', ',');
                            string heapSortTime = results[i, 2].ToString("F8").Replace('.', ',');

                            // Используем точку с запятой как разделитель колонок 
                            sw.WriteLine($"{results[i, 0]};{insertionSortTime};{heapSortTime}");
                        }
                    }

                    Log($"Сохранено в {dlg.FileName}");
                    MessageBox.Show($"Результаты сохранены!\n{dlg.FileName}",
                        "Сохранение", MessageBoxButtons.OK, MessageBoxIcon.Information);
                }
                //если при сохранении ошибка
                catch (Exception ex)
                {
                    MessageBox.Show($"Ошибка при сохранении: {ex.Message}",
                        "Ошибка", MessageBoxButtons.OK, MessageBoxIcon.Error);
                    Log($"Ошибка сохранения: {ex.Message}");
                }
            }
        }

        private void BtnLoad_Click(object sender, EventArgs e)
        {
            // диалоговаой окно для открытия файла
            OpenFileDialog dlg = new OpenFileDialog();
            dlg.Filter = "CSV файлы (*.csv)|*.csv|Текстовые файлы (*.txt)|*.txt|Все файлы (*.*)|*.*";

            if (dlg.ShowDialog() == DialogResult.OK)
            {
                try
                {
                    string[] fileData = File.ReadAllLines(dlg.FileName);
                    dgvResults.Rows.Clear();

                    if (fileData.Length == 0)
                    {
                        MessageBox.Show("Файл пуст!", "Ошибка", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                        return;
                    }

                    Log($"Начинаем загрузку файла: {dlg.FileName}");
                    Log($"Всего строк в файле: {fileData.Length}");

                    int loadedCount = 0;
                    int errorCount = 0;

                    // загрузка строк
                    for (int i = 0; i < fileData.Length; i++)
                    {
                        string line = fileData[i].Trim();
                        if (string.IsNullOrEmpty(line)) continue;

                        Log($"Обработка строки {i + 1}: '{line}'");

                        // Определяем разделитель
                        char separator = ',';
                        if (line.Contains(';')) separator = ';';
                        else if (line.Contains('\t')) separator = '\t';

                        string[] parts = line.Split(separator);

                        // Пропускаем заголовок (если есть слова "n", "insertion", "heap")
                        if (i == 0)
                        {
                            string lowerLine = line.ToLower();
                            if (lowerLine.Contains("n") && (lowerLine.Contains("insertion") || lowerLine.Contains("heap")))
                            {
                                Log("Пропускаем заголовок");
                                continue;
                            }
                        }

                        if (parts.Length < 3)
                        {
                            Log($"  ОШИБКА: недостаточно колонок (найдено {parts.Length})");
                            errorCount++;
                            continue;
                        }



                        string countStr = parts[0].Trim();
                        string insertionSortTime = parts[1].Trim();
                        string heapSortTime = parts[2].Trim();

                        Log($"  n='{countStr}', вставки='{insertionSortTime}', пирамида='{heapSortTime}'");

                        // Преобразуем запятую в точку для десятичных чисел
                        insertionSortTime = insertionSortTime.Replace(',', '.');
                        heapSortTime = heapSortTime.Replace(',', '.');

                        // Добавляем в таблицу
                        dgvResults.Rows.Add(countStr, insertionSortTime, heapSortTime);
                        loadedCount++;

                        // Сохраняем в массив results (для дальнейшего использования)
                        if (loadedCount <= N_VALUES.Length)
                        {
                            int n = 0;
                            int.TryParse(countStr, out n);

                            double t1 = 0, t2 = 0;
                            double.TryParse(insertionSortTime, System.Globalization.NumberStyles.Any,
                                System.Globalization.CultureInfo.InvariantCulture, out t1);
                            double.TryParse(heapSortTime, System.Globalization.NumberStyles.Any,
                                System.Globalization.CultureInfo.InvariantCulture, out t2);

                            results[loadedCount - 1, 0] = n;
                            results[loadedCount - 1, 1] = t1;
                            results[loadedCount - 1, 2] = t2;
                        }
                    }

                    Log($"Загрузка завершена: загружено {loadedCount} записей, ошибок {errorCount}");

                    if (loadedCount == 0)
                    {
                        MessageBox.Show("Не удалось загрузить данные из файла.\nПроверьте формат файла.",
                            "Ошибка", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                    }
                    else
                    {
                        MessageBox.Show($"Загружено {loadedCount} записей!",
                            "Загрузка", MessageBoxButtons.OK, MessageBoxIcon.Information);
                        btnSave.Enabled = true;
                    }
                }
                catch (Exception ex)
                {
                    MessageBox.Show($"Ошибка при загрузке: {ex.Message}",
                        "Ошибка", MessageBoxButtons.OK, MessageBoxIcon.Error);
                    Log($"Ошибка загрузки: {ex.Message}");
                }
            }
        }

        private void Form1_Load(object sender, EventArgs e)
        {

        }
    }
}