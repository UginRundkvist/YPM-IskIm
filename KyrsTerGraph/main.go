<<<<<<< HEAD
// примерно 5500 символов k мер перекрытие+1
//     Бактериофаг phiX174 — идеальный вариант! Это небольшой вирус (~5,386 нуклеотидов) m = 4(отрезки для оценуи сложности), его геном полностью секвенирован и хорошо изучен
// E. coli K-12 — если хотите побольше (~4.6 млн нуклеотидов)  m = 6, доступен полностью собранный геном

//Параметр	Значение для 5500 
//Длина рида	50 bp
//Перекрытие	25 bp (50%)
//k-мер	26
//Глубина покрытия	10×

//Геном Б (длинный) — 4,5 млн bp
//Параметр	Значение
//Длина рида	150 bp
//Перекрытие	90 bp (60%)
//k-мер	91
//Глубина покрытия	30×

package main

import (
    "fmt"
    "math/rand"
    "time"
)

// Config содержит параметры сборки
type Config struct {
    ReadLength int // длина рида в bp
    Overlap    int // перекрытие между ридами в bp
    K          int // размер k-мера (Overlap + 1)
    Coverage   int // глубина покрытия
}

// Предустановка для короткого генома (5 500 bp)
func ShortConfig() Config {
    return Config{
        ReadLength: 50,
        Overlap:    25,
        K:          26,
        Coverage:   10,
    }
}

// Предустановка для длинного генома (4,5 млн bp)
func LongConfig() Config {
    return Config{
        ReadLength: 150,
        Overlap:    90,
        K:          91,
        Coverage:   30,
    }
}

// GenerateReads нарезает геном на риды с заданными параметрами
func GenerateReads(genome string, cfg Config) []string {
    var reads []string
    
    step := cfg.ReadLength - cfg.Overlap // шаг между началами ридов
    
    if step <= 0 {
        fmt.Println("Ошибка: перекрытие больше или равно длине рида")
        return reads
    }
    
    // Проходим с разными сдвигами для достижения нужной глубины покрытия
    for shift := 0; shift < cfg.Coverage; shift++ {
        startPos := shift
        
        for startPos+cfg.ReadLength <= len(genome) {
            read := genome[startPos : startPos+cfg.ReadLength]
            reads = append(reads, read)
            startPos += step
        }
    }
    
    return reads
}

// GenerateRandomDNA создает случайную ДНК заданной длины
func GenerateRandomDNA(length int) string {
    bases := []byte{'A', 'C', 'G', 'T'}
    rand.Seed(time.Now().UnixNano())
    
    dna := make([]byte, length)
    for i := 0; i < length; i++ {
        dna[i] = bases[rand.Intn(4)]
    }
    return string(dna)
}

// PrintStats выводит статистику по сгенерированным ридам
func PrintStats(reads []string, genomeLength int, cfg Config) {
    fmt.Println("=== Статистика ридов ===")
    fmt.Printf("Длина генома: %d bp\n", genomeLength)
    fmt.Printf("Длина рида: %d bp\n", cfg.ReadLength)
    fmt.Printf("Перекрытие: %d bp (%.0f%%)\n", cfg.Overlap, float64(cfg.Overlap)/float64(cfg.ReadLength)*100)
    fmt.Printf("Размер k-мера: %d\n", cfg.K)
    fmt.Printf("Глубина покрытия: %d×\n", cfg.Coverage)
    fmt.Printf("Количество ридов: %d\n", len(reads))
    
    // Оценка реальной глубины покрытия
    expectedReadsCount := (genomeLength / (cfg.ReadLength - cfg.Overlap)) * cfg.Coverage
    fmt.Printf("Ожидаемое количество ридов: ~%d\n", expectedReadsCount)
}

func main() {
    // === ТЕСТ 1: КОРОТКИЙ ГЕНОМ ===
    fmt.Println("====== ТЕСТ 1: Короткий геном ======")
    
    // Создаем тестовую ДНК длиной 500 bp (для демонстрации)
    shortGenome := GenerateRandomDNA(500)
    cfgShort := ShortConfig()
    
    readsShort := GenerateReads(shortGenome, cfgShort)
    PrintStats(readsShort, len(shortGenome), cfgShort)
    
    // Показываем первые 5 ридов
    fmt.Println("\nПримеры ридов (первые 5):")
    for i := 0; i < 5 && i < len(readsShort); i++ {
        fmt.Printf("  %d: %s\n", i+1, readsShort[i])
    }
    
    // === ТЕСТ 2: ДЛИННЫЙ ГЕНОМ ===
    fmt.Println("\n====== ТЕСТ 2: Длинный геном ======")
    
    // Для теста берем 5000 bp
    longGenome := GenerateRandomDNA(5000)
    cfgLong := LongConfig()
    
    readsLong := GenerateReads(longGenome, cfgLong)
    PrintStats(readsLong, len(longGenome), cfgLong)
    
    fmt.Println("\nПримеры ридов (первые 5):")
    for i := 0; i < 5 && i < len(readsLong); i++ {
        fmt.Printf("  %d: %s\n", i+1, readsLong[i])
    }
}
=======
package main

import (
	"bufio"
	"flag"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

type Config struct {
	GenomePath string  // путь к FASTA/fna файлу с геномом
	ReadLen    int     // длина рида
	Overlap    int     // перекрытие (для k = overlap+1)
	Coverage   float64 // глубина покрытия
	K          int     // k-мер (будет вычислен как overlap+1)
	OutputDir  string  // директория для выходных файлов
}

// Загрузка файла
func loadGenomeFromFile(path string) (string, error) {
	file, err := os.Open(path)
	if err != nil {
		return "", fmt.Errorf("не удалось открыть файл %s: %v", path, err)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	var genome strings.Builder
	lineNum := 0
	sequenceStarted := false

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		lineNum++
		if line == "" {
			continue
		}

		if line[0] == '>' {
			sequenceStarted = true
			fmt.Printf("  Заголовок FASTA/fna (строка %d): %s\n", lineNum,
				line[:min(60, len(line))])
			if len(line) > 60 {
				fmt.Print("...\n")
			}
			continue
		}

		if !sequenceStarted && lineNum == 1 {
			fmt.Println("  Файл не содержит заголовка '>', читаем как последовательность")
			sequenceStarted = true
		}

		cleanLine := strings.Map(func(r rune) rune {
			if r >= 'A' && r <= 'Z' || r >= 'a' && r <= 'z' {
				return r
			}
			return -1
		}, line)

		if len(cleanLine) > 0 {
			genome.WriteString(strings.ToUpper(cleanLine))
		}
	}

	if err := scanner.Err(); err != nil {
		return "", fmt.Errorf("ошибка чтения файла: %v", err)
	}

	result := genome.String()
	if len(result) == 0 {
		return "", fmt.Errorf("файл %s не содержит последовательностей ДНК", path)
	}

	// Подсчитываем количество каждого нуклеотида
	stats := map[byte]int{'A': 0, 'T': 0, 'G': 0, 'C': 0, 'N': 0}
	for i := 0; i < len(result); i++ {
		c := result[i]
		if _, ok := stats[c]; ok {
			stats[c]++
		} else {
			stats['N']++
		}
	}

	fmt.Printf("\n  Загружен геном из %s:\n", path)
	fmt.Printf("    Длина: %d нуклеотидов\n", len(result))
	fmt.Printf("    Состав: A=%d (%.1f%%), T=%d (%.1f%%), G=%d (%.1f%%), C=%d (%.1f%%)\n",
		stats['A'], float64(stats['A'])/float64(len(result))*100,
		stats['T'], float64(stats['T'])/float64(len(result))*100,
		stats['G'], float64(stats['G'])/float64(len(result))*100,
		stats['C'], float64(stats['C'])/float64(len(result))*100)

	if stats['N'] > 0 {
		fmt.Printf("    Неопределённых (N): %d (%.2f%%)\n",
			stats['N'], float64(stats['N'])/float64(len(result))*100)
	}

	return result, nil
}

// Генерация ридов
func generateReads(genome string, readLen int, coverage float64) []string {

	genomeLen := len(genome)
	if genomeLen < readLen {
		fmt.Printf("  Предупреждение: геном (%d bp) короче длины рида (%d bp)\n", genomeLen, readLen)
		return []string{genome}
	}

	// Количество ридов для заданного покрытия
	targetReads := int(float64(genomeLen) * coverage / float64(readLen))
	if targetReads < 1 {
		targetReads = 1
	}

	reads := make([]string, 0, targetReads)
	//rand.Seed(time.Now().UnixNano())

	for len(reads) < targetReads {
		start := rand.Intn(genomeLen - readLen + 1)
		read := genome[start : start+readLen]
		reads = append(reads, read)
	}

	fmt.Printf("\n  Сгенерировано %d ридов:\n", len(reads))
	fmt.Printf("    Длина рида: %d bp\n", readLen)
	fmt.Printf("    Покрытие: %.1fx\n", coverage)
	fmt.Printf("    Ожидаемое покрытие: %.1fx\n",
		float64(len(reads)*readLen)/float64(genomeLen))

	return reads
}

func saveReadsToFasta(reads []string, outputPath string) error {
	file, err := os.Create(outputPath)
	if err != nil {
		return err
	}
	defer file.Close()

	writer := bufio.NewWriter(file)
	for i, read := range reads {
		fmt.Fprintf(writer, ">read_%d\n", i+1)
		for j := 0; j < len(read); j += 80 {
			end := j + 80
			if end > len(read) {
				end = len(read)
			}
			fmt.Fprintf(writer, "%s\n", read[j:end])
		}
	}
	writer.Flush()
	fmt.Printf("\n  Риды сохранены в %s\n", outputPath)
	return nil
}

type DeBruijnGraph struct {
	Edges   map[string][]string // левый узел -> список правых узлов
	Weights map[string]int      // вес ребра (счётчик)
	Nodes   map[string]bool     // все уникальные узлы
}

func NewDeBruijnGraph(k int, reads []string) *DeBruijnGraph {
	graph := &DeBruijnGraph{
		Edges:   make(map[string][]string),
		Weights: make(map[string]int),
		Nodes:   make(map[string]bool),
	}

	if k < 2 {
		fmt.Println("  Ошибка: k должно быть >= 2")
		return graph
	}

	totalKmers := 0
	for _, read := range reads {
		for i := 0; i <= len(read)-k; i++ {
			kmer := read[i : i+k]
			left := kmer[:k-1]
			right := kmer[1:]

			graph.Nodes[left] = true
			graph.Nodes[right] = true

			edgeKey := fmt.Sprintf("%s|%s", left, right)
			graph.Weights[edgeKey]++

			// Добавляем ребро, если его ещё нет
			found := false
			for _, existing := range graph.Edges[left] {
				if existing == right {
					found = true
					break
				}
			}
			if !found {
				graph.Edges[left] = append(graph.Edges[left], right)
			}
			totalKmers++
		}
	}

	fmt.Printf("\n  Построен граф де Брюйна:\n")
	fmt.Printf("    k = %d\n", k)
	fmt.Printf("    Узлов (k-1-меров): %d\n", len(graph.Nodes))
	fmt.Printf("    Рёбер (уникальных): %d\n", len(graph.Edges))
	fmt.Printf("    Всего k-меров: %d\n", totalKmers)

	// Плотность графа
	if len(graph.Nodes) > 0 {
		density := float64(len(graph.Edges)) / float64(len(graph.Nodes))
		fmt.Printf("    Плотность рёбер: %.2f\n", density)
	}

	return graph
}

// Вычисление степеней узлов
func (g *DeBruijnGraph) CalculateDegrees() (inDegree, outDegree map[string]int) {
	inDegree = make(map[string]int)
	outDegree = make(map[string]int)

	for left, rights := range g.Edges {
		outDegree[left] = len(rights)
		for _, right := range rights {
			inDegree[right]++
		}
	}

	// Для узлов, которые есть только как правые
	for node := range g.Nodes {
		if _, ok := outDegree[node]; !ok {
			outDegree[node] = 0
		}
		if _, ok := inDegree[node]; !ok {
			inDegree[node] = 0
		}
	}
	return inDegree, outDegree
}

// Анализ
func (g *DeBruijnGraph) Analyze() {
	inDeg, outDeg := g.CalculateDegrees()

	// Статистика по степеням
	branching := 0 // узлы с ветвлением
	terminals := 0 // тупиковые узлы
	linear := 0    // линейные узлы
	ideal := 0     // идеальные узлы

	for node := range g.Nodes {
		in := inDeg[node]
		out := outDeg[node]

		if in == 0 && out == 0 {
			terminals++
		} else if in > 1 || out > 1 {
			branching++
		} else if in == 1 && out == 1 {
			ideal++
		} else {
			linear++
		}
	}

	fmt.Printf("\n  Анализ структуры графа:\n")
	fmt.Printf("    Идеальные узлы : %d (%.1f%%)\n",
		ideal, float64(ideal)/float64(len(g.Nodes))*100)
	fmt.Printf("    Линейные узлы: %d (%.1f%%)\n",
		linear, float64(linear)/float64(len(g.Nodes))*100)
	fmt.Printf("    Узлы с ветвлением: %d (%.1f%%)\n",
		branching, float64(branching)/float64(len(g.Nodes))*100)
	fmt.Printf("    Тупиковые узлы: %d (%.1f%%)\n",
		terminals, float64(terminals)/float64(len(g.Nodes))*100)

	if branching > len(g.Nodes)/10 {
		fmt.Printf("    ⚠️  Высокий уровень ветвления — возможны повторы в геноме\n")
	}
}

// Перепроверть
func (g *DeBruijnGraph) Assemble() []string {
	if len(g.Edges) == 0 {
		return []string{}
	}

	inDeg, outDeg := g.CalculateDegrees()

	// Находим стартовые узлы (out > in) и узлы без входящих рёбер
	startNodes := []string{}
	for node := range g.Nodes {
		if outDeg[node] > inDeg[node] || (inDeg[node] == 0 && outDeg[node] > 0) {
			startNodes = append(startNodes, node)
		}
	}

	// Если нет подходящих стартовых узлов, начинаем с узлов с максимальным out-degree
	if len(startNodes) == 0 {
		maxOut := 0
		for _, out := range outDeg {
			if out > maxOut {
				maxOut = out
			}
		}
		for node, out := range outDeg {
			if out == maxOut && out > 0 {
				startNodes = append(startNodes, node)
			}
		}
	}

	// Сортируем стартовые узлы
	sort.Slice(startNodes, func(i, j int) bool {
		return outDeg[startNodes[i]] > outDeg[startNodes[j]]
	})

	contigs := []string{}
	usedEdges := make(map[string]bool)

	for _, start := range startNodes {
		if outDeg[start] == 0 {
			continue
		}

		contig := start
		current := start
		pathLength := 0
		maxPathLength := len(g.Nodes) * 2 // Защита от бесконечных циклов

		for pathLength < maxPathLength {
			neighbors, ok := g.Edges[current]
			if !ok || len(neighbors) == 0 {
				break
			}

			// Выбираем непосещённое ребро с максимальным весом
			best := ""
			bestWeight := -1
			for _, nb := range neighbors {
				edgeKey := fmt.Sprintf("%s|%s", current, nb)
				if usedEdges[edgeKey] {
					continue
				}
				if w, ok := g.Weights[edgeKey]; ok && w > bestWeight {
					bestWeight = w
					best = nb
				}
			}

			if best == "" {
				break
			}

			edgeKey := fmt.Sprintf("%s|%s", current, best)
			usedEdges[edgeKey] = true

			// Добавляем последний символ best
			contig += string(best[len(best)-1])
			current = best
			pathLength++
		}

		if len(contig) > 0 && len(contig) >= 100 { // Минимальная длина контига
			contigs = append(contigs, contig)
		}
	}

	// Сортируем контиги по длине (убывание)
	sort.Slice(contigs, func(i, j int) bool {
		return len(contigs[i]) > len(contigs[j])
	})

	fmt.Printf("\n  Собрано %d контигов (минимальная длина 100 bp)\n", len(contigs))
	if len(contigs) > 0 {
		fmt.Printf("    Самый длинный: %d bp\n", len(contigs[0]))
	}

	return contigs
}

func saveContigsToFile(contigs []string, outputPath string) error {
	file, err := os.Create(outputPath)
	if err != nil {
		return err
	}
	defer file.Close()

	writer := bufio.NewWriter(file)
	for i, contig := range contigs {
		fmt.Fprintf(writer, ">contig_%d length=%d\n", i+1, len(contig))
		for j := 0; j < len(contig); j += 80 {
			end := j + 80
			if end > len(contig) {
				end = len(contig)
			}
			fmt.Fprintf(writer, "%s\n", contig[j:end])
		}
	}
	writer.Flush()
	fmt.Printf("\n  Контиги сохранены в %s\n", outputPath)
	return nil
}

// Статистика
type AssemblyStats struct {
	NumContigs     int
	TotalLength    int
	MaxContig      int
	MinContig      int
	N50            int
	GenomeCoverage float64
}

func calculateStats(contigs []string, genomeLength int) AssemblyStats {
	if len(contigs) == 0 {
		return AssemblyStats{GenomeCoverage: 0}
	}

	lengths := make([]int, len(contigs))
	total := 0
	maxLen := 0
	minLen := int(^uint(0) >> 1)

	for i, c := range contigs {
		l := len(c)
		lengths[i] = l
		total += l
		if l > maxLen {
			maxLen = l
		}
		if l < minLen {
			minLen = l
		}
	}

	//N50 — это длина такого контига, что все контиги длиннее или равные ему покрывают ≥50% всего генома
	sort.Sort(sort.Reverse(sort.IntSlice(lengths)))
	half := total / 2
	sum := 0
	n50 := 0
	for _, l := range lengths {
		sum += l
		if sum >= half {
			n50 = l
			break
		}
	}

	return AssemblyStats{
		NumContigs:     len(contigs),
		TotalLength:    total,
		MaxContig:      maxLen,
		MinContig:      minLen,
		N50:            n50,
		GenomeCoverage: float64(total) / float64(genomeLength) * 100.0,
	}
}

func (s AssemblyStats) String() string {
	return fmt.Sprintf(`
СТАТИСТИКА СБОРКИ
Количество контигов:     %d
Общая длина:             %d bp
Самый длинный контиг:    %d bp
Самый короткий контиг:   %d bp
N50:                     %d bp
Покрытие генома:         %.2f%% `,
		s.NumContigs, s.TotalLength, s.MaxContig, s.MinContig, s.N50, s.GenomeCoverage)
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func main() {
	// Параметры командной строки
	genomeFile := flag.String("genome", "genomes/phiX174.fna", "Путь к FASTA/fna файлу с геномом")
	readLen := flag.Int("readlen", 50, "Длина рида (bp)")
	overlap := flag.Int("overlap", 25, "Перекрытие между ридами (bp)")
	coverage := flag.Float64("coverage", 10.0, "Глубина покрытия (x)")
	outputDir := flag.String("output", "output", "Директория для выходных файлов")
	flag.Parse()

	// Проверяем корректность параметров
	if *overlap >= *readLen {
		fmt.Println("Ошибка: перекрытие должно быть меньше длины рида")
		return
	}

	k := *overlap + 1

	fmt.Printf(`
Параметры эксперимента:
   Геном:               %s
   Длина рида:          %d bp
   Перекрытие:          %d bp
   k-мер:               %d
   Покрытие:            %.1fx
   Выходная директория: %s
`,
		*genomeFile, *readLen, *overlap, k, *coverage, *outputDir)

	// Создаём выходную директорию
	if err := os.MkdirAll(*outputDir, 0755); err != nil {
		fmt.Printf(" Ошибка создания директории: %v\n", err)
		return
	}

	// Загружаем геном
	fmt.Println("\n[1]  Загрузка генома...")
	genome, err := loadGenomeFromFile(*genomeFile)
	if err != nil {
		fmt.Printf(" Ошибка: %v\n", err)
		return
	}

	//  Генерируем риды
	fmt.Println("\n[2]  Генерация ридов...")
	reads := generateReads(genome, *readLen, *coverage)

	// Сохраняем риды
	readsFile := filepath.Join(*outputDir, "reads.fasta")
	if err := saveReadsToFasta(reads, readsFile); err != nil {
		fmt.Printf("  Предупреждение: не удалось сохранить риды: %v\n", err)
	}

	// Строим граф де Брюйна
	fmt.Println("\n[3]   Построение графа де Брюйна...")
	graph := NewDeBruijnGraph(k, reads)

	// Анализ графа
	graph.Analyze()

	//  Сборка контигов
	fmt.Println("\n[4]  Сборка контигов...")
	contigs := graph.Assemble()

	// Статистика и сохранение
	fmt.Println("\n[5]  Сохранение результатов...")

	// Сохраняем контиги
	contigsFile := filepath.Join(*outputDir, "contigs.fna")
	if err := saveContigsToFile(contigs, contigsFile); err != nil {
		fmt.Printf(" Ошибка сохранения контигов: %v\n", err)
	}

	// Статистика
	stats := calculateStats(contigs, len(genome))
	fmt.Println(stats.String())

	// Сохраняем статистику
	statsFile := filepath.Join(*outputDir, "stats.txt")
	if f, err := os.Create(statsFile); err == nil {
		defer f.Close()
		fmt.Fprintf(f, "=== ОТЧЁТ О СБОРКЕ ГЕНОМА ===\n\n")
		fmt.Fprintf(f, "Исходные данные:\n")
		fmt.Fprintf(f, "  Файл генома: %s\n", *genomeFile)
		fmt.Fprintf(f, "  Длина генома: %d bp\n", len(genome))
		fmt.Fprintf(f, "  Длина рида: %d bp\n", *readLen)
		fmt.Fprintf(f, "  Перекрытие: %d bp\n", *overlap)
		fmt.Fprintf(f, "  k-мер: %d\n", k)
		fmt.Fprintf(f, "  Покрытие: %.1fx\n", *coverage)
		fmt.Fprintf(f, "  Количество ридов: %d\n", len(reads))
		fmt.Fprintf(f, "\n%s\n", stats.String())
		fmt.Fprintf(f, "\nДетали графа:\n")
		fmt.Fprintf(f, "  Узлов: %d\n", len(graph.Nodes))
		fmt.Fprintf(f, "  Рёбер: %d\n", len(graph.Edges))

		fmt.Printf("\n   Статистика сохранена в %s\n", statsFile)
	}
}

//для phiX174
// go run main.go -genome genomes/phiX174.fna -readlen 50 -overlap 25 -coverage 10

//  для  ecoli
// go run main.go -genome genomes/ecoli.fna -readlen 150 -overlap 90 -coverage 30
>>>>>>> 8f13fed (heheh)
