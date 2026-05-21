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