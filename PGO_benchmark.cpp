#include <iostream>
#include <vector>
#include <string>
#include <thread>
#include <mutex>
#include <numeric>
#include <algorithm>
#include <map>
#include <list>
#include <stdexcept>
#include <functional>
#include <atomic>
#include <queue>    // For Dijkstra's
#include <limits>   // For std::numeric_limits
#include <cctype>   // For isalpha, tolower

// System headers
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <signal.h>

// --- Global state for threading nightmares ---
std::mutex mutex_a;
std::mutex mutex_b;
long shared_racy_data = 0;

// ====================================================================
// PART 1: MEMORY ABUSER - Triggers for Valgrind's Memcheck
// ====================================================================
namespace MemoryAbuser {
void trigger_use_after_free() {
    std::cout << "[Mem] Triggering use-after-free..." << std::endl;
    int* ptr = new int[10];
    ptr[5] = 123;
    delete[] ptr;
    // ERROR: Using freed memory. Valgrind will detect this read.
    std::cout << "[Mem] Value from freed memory: " << ptr[5] << std::endl;
}

void trigger_heap_overflow() {
    std::cout << "[Mem] Triggering heap buffer overflow..." << std::endl;
    char* buffer = new char[16];
    // ERROR: Writing 1 byte past the allocated chunk.
    buffer[16] = 'X';
    std::cout << "[Mem] Overflowed with char (read may be invalid): " << buffer[16] << std::endl;
    delete[] buffer;
}

void trigger_uninitialized_read() {
    std::cout << "[Mem] Triggering uninitialized value read..." << std::endl;
    int x; // Uninitialized
    // ERROR: x is uninitialized and used in a conditional jump.
    if (x > 100) {
        x = 1;
    } else {
        x = 0;
    }
    std::cout << "[Mem] Uninitialized value was used in a branch." << std::endl;
}

void trigger_memory_leak() {
    std::cout << "[Mem] Triggering a memory leak..." << std::endl;
    // ERROR: Leaking this memory.
    new std::string("This string is a memory leak.");
}

void run() {
    std::cout << "\n--- Running Memory Abuser ---" << std::endl;
    trigger_use_after_free();
    trigger_heap_overflow();
    trigger_uninitialized_read();
    trigger_memory_leak();
}
} // namespace MemoryAbuser


// ====================================================================
// PART 2: THREADING NIGHTMARE - Data races & deadlocks for Helgrind/DRD
// ====================================================================
namespace ThreadingNightmare {
void racy_thread_func() {
    for (int i = 0; i < 10000; ++i) {
        // ERROR: Data race. No lock protecting the shared data.
        shared_racy_data++;
    }
}

void deadlock_thread_a() {
    std::cout << "[Thread] Deadlock thread A starting..." << std::endl;
    std::lock_guard<std::mutex> lock_a(mutex_a);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    std::cout << "[Thread] A trying to get lock B..." << std::endl;
    std::lock_guard<std::mutex> lock_b(mutex_b); // Will likely deadlock here
}

void deadlock_thread_b() {
    std::cout << "[Thread] Deadlock thread B starting..." << std::endl;
    std::lock_guard<std::mutex> lock_b(mutex_b);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    std::cout << "[Thread] B trying to get lock A..." << std::endl;
    std::lock_guard<std::mutex> lock_a(mutex_a); // Will likely deadlock here
}

void run() {
    std::cout << "\n--- Running Threading Nightmare ---" << std::endl;
    std::vector<std::thread> threads;

    std::cout << "[Thread] Spawning threads for data race..." << std::endl;
    for (int i = 0; i < 4; ++i) threads.emplace_back(racy_thread_func);
    for (auto& t : threads) t.join();
    threads.clear();
    std::cout << "[Thread] Final racy value: " << shared_racy_data << std::endl;

    std::cout << "[Thread] Spawning threads for deadlock (may hang)..." << std::endl;
    threads.emplace_back(deadlock_thread_a);
    threads.emplace_back(deadlock_thread_b);
    for (auto& t : threads) t.join();
}
} // namespace ThreadingNightmare

// ====================================================================
// PART 3: IPC & SYSCALL CHAOS - fork, pipes, fds, mmap
// ====================================================================
namespace SyscallChaos {
void run() {
    std::cout << "\n--- Running IPC and Syscall Chaos ---" << std::endl;
    int pipe_fd[2];
    if (pipe(pipe_fd) == -1) { perror("pipe"); return; }

    pid_t pid = fork();
    if (pid == -1) { perror("fork"); return; }

    if (pid == 0) { // Child
        close(pipe_fd[0]);
        std::string msg = "Message from child across the pipe!";
        write(pipe_fd[1], msg.c_str(), msg.length() + 1);
        close(pipe_fd[1]);

        void* mem = mmap(NULL, 4096, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (mem != MAP_FAILED) {
            strcpy((char*)mem, "mmap successful");
            munmap(mem, 4096);
        }
        _exit(0);
    } else { // Parent
        close(pipe_fd[1]);
        char buffer[128] = {0};
        read(pipe_fd[0], buffer, sizeof(buffer) - 1);
        std::cout << "[IPC] Parent received: '" << buffer << "'" << std::endl;
        close(pipe_fd[0]);
        wait(NULL);
    }

    std::cout << "[Syscall] Leaking a file descriptor..." << std::endl;
    // ERROR: This file descriptor will be leaked.
    (void)open("/dev/null", O_RDONLY);
}
} // namespace SyscallChaos

// ====================================================================
// PART 4: ADVANCED ALGORITHM STRESS TEST
// ====================================================================
namespace AlgorithmStress {

// --- 4a: Graph algorithm (Dijkstra's) ---
void run_dijkstra() {
    std::cout << "[Algo] Running Dijkstra's Algorithm..." << std::endl;
    int num_vertices = 6;
    using Edge = std::pair<int, int>; // {weight, destination}
    std::vector<std::list<Edge>> adj(num_vertices);

    // A sample graph
    adj[0].push_back({4, 1}); adj[0].push_back({2, 2});
    adj[1].push_back({5, 2}); adj[1].push_back({10, 3});
    adj[2].push_back({3, 4});
    adj[3].push_back({7, 5}); adj[3].push_back({1, 4});
    adj[4].push_back({8, 5});

    std::priority_queue<Edge, std::vector<Edge>, std::greater<Edge>> pq;
    std::vector<int> dist(num_vertices, std::numeric_limits<int>::max());

    int start_node = 0;
    pq.push({0, start_node});
    dist[start_node] = 0;

    while (!pq.empty()) {
        int u = pq.top().second;
        pq.pop();

        for (const auto& edge : adj[u]) {
            int v = edge.second;
            int weight = edge.first;
            if (dist[u] + weight < dist[v]) {
                dist[v] = dist[u] + weight;
                pq.push({dist[v], v});
            }
        }
    }
    // This complex loop structure and data access is great for PGO
}

// --- 4b: Recursive Backtracking (N-Queens) ---
namespace NQueens {
    bool is_safe(int row, int col, const std::vector<int>& board) {
        for (int prev_row = 0; prev_row < row; ++prev_row) {
            int prev_col = board[prev_row];
            if (prev_col == col || std::abs(prev_row - row) == std::abs(prev_col - col)) {
                return false;
            }
        }
        return true;
    }

    int solve(int row, std::vector<int>& board) {
        int n = board.size();
        if (row == n) {
            return 1; // Found one solution
        }
        int count = 0;
        for (int col = 0; col < n; ++col) {
            if (is_safe(row, col, board)) {
                board[row] = col;
                count += solve(row + 1, board);
            }
        }
        return count;
    }
}
void run_n_queens() {
    std::cout << "[Algo] Running N-Queens Solver (N=9)..." << std::endl;
    int n = 9; // Challenging but not too slow
    std::vector<int> board(n);
    NQueens::solve(0, board); // Stresses call stack
}

// --- 4c: Heavy String & Map Usage (Word Frequency) ---
void run_word_frequency() {
    std::cout << "[Algo] Running Word Frequency Counter..." << std::endl;
    const std::string text = R"(
        Valgrind is an instrumentation framework for building dynamic analysis tools.
        There are Valgrind tools that can automatically detect many memory management
        and threading bugs, and profile your programs in detail. The Valgrind
        distribution also includes a tool that can build new Valgrind tools.
        This process of instrumentation is complex. And this text will be parsed.
        The most common use-case for Valgrind is memory checking, but the profiler
        is also extremely useful. Profiling provides data for optimization.
    )";
    
    std::map<std::string, int> frequencies;
    std::string current_word;
    for (char c : text) {
        if (std::isalpha(c)) {
            current_word += std::tolower(c);
        } else {
            if (!current_word.empty()) {
                frequencies[current_word]++;
                current_word.clear();
            }
        }
    }
    if (!current_word.empty()) {
        frequencies[current_word]++;
    }
    // Stresses map operations, string allocations, and character logic.
}

// --- 4d: Numerical Computation (Matrix Multiplication) ---
void run_matrix_multiplication() {
    std::cout << "[Algo] Running Matrix Multiplication..." << std::endl;
    int size = 50;
    std::vector<std::vector<int>> a(size, std::vector<int>(size));
    std::vector<std::vector<int>> b(size, std::vector<int>(size));
    std::vector<std::vector<int>> result(size, std::vector<int>(size, 0));

    // Initialize matrices
    for(int i=0; i<size; ++i) {
        for(int j=0; j<size; ++j) {
            a[i][j] = i + j;
            b[i][j] = i - j;
        }
    }

    // Triple nested loop is a classic hot spot for PGO to optimize
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            for (int k = 0; k < size; ++k) {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }
}

void run() {
    std::cout << "\n--- Running Advanced Algorithm Stress Test ---" << std::endl;
    run_dijkstra();
    run_n_queens();
    run_word_frequency();
    run_matrix_multiplication();
}
} // namespace AlgorithmStress

// ====================================================================
// --- MAIN ORCHESTRATOR ---
// ====================================================================
int main() {
    std::cout << "Starting ultimate Valgrind stress test..." << std::endl;

    // Run all tests sequentially.
    MemoryAbuser::run();
    SyscallChaos::run();
    AlgorithmStress::run();

    // The threading test is last because it may deadlock and hang,
    // preventing other tests from running.
    ThreadingNightmare::run();

    std::cout << "\nUltimate stress test finished." << std::endl;
    return 0;
}
