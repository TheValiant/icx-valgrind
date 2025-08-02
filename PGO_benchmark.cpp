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
#include <queue>
#include <limits>
#include <cctype>
#include <cstring> // For C-style string functions
#include <chrono>  // For timing
#include <condition_variable>
#include <string.h>

// System headers
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <signal.h>

// --- Global state for threading nightmares ---
std::mutex g_mutex_a;
std::mutex g_mutex_b;
long g_shared_racy_data = 0;

// ====================================================================
// PART 1: MEMORY ABUSER - Triggers for Valgrind's Memcheck
// ====================================================================
namespace MemoryAbuser {

void trigger_use_after_free() {
    std::cout << "[Mem] Triggering use-after-free..." << std::endl;
    int* ptr = new int[50];
    ptr[25] = 123;
    delete[] ptr;
    // ERROR: Using freed memory. Valgrind will detect this read.
    // We suppress the output as it might crash, but Valgrind sees the read.
    volatile int val = ptr[25]; 
    (void)val; // Prevent compiler from optimizing away the read.
}

void trigger_heap_overflow() {
    std::cout << "[Mem] Triggering heap buffer overflow..." << std::endl;
    char* buffer = new char[32];
    // ERROR: Writing 1 byte past the allocated chunk.
    buffer[32] = 'X'; 
    delete[] buffer;
}

void trigger_stack_overflow() {
    std::cout << "[Mem] Triggering stack buffer overflow..." << std::endl;
    char buffer[16];
    // ERROR: Writing past the end of a stack-allocated buffer.
    strcpy(buffer, "This string is way too long for the buffer!");
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
    std::cout << "[Mem] Uninitialized value was used in a branch (result: " << x << ")." << std::endl;
}

void trigger_memory_leak() {
    std::cout << "[Mem] Triggering a definite memory leak..." << std::endl;
    // ERROR: Leaking this memory.
    new std::string("This string is a definite memory leak.");
}

void trigger_mismatched_free() {
    std::cout << "[Mem] Triggering mismatched new[]/delete..." << std::endl;
    // ERROR: Allocating an array but using scalar delete.
    int* array = new int[10];
    delete array;
}

void trigger_double_free() {
    std::cout << "[Mem] Triggering double free..." << std::endl;
    int* p = new int(42);
    delete p;
    // ERROR: Deleting the same memory twice.
    delete p;
}

void trigger_overlapping_memcpy() {
    std::cout << "[Mem] Triggering overlapping src/dst in memcpy..." << std::endl;
    char data[20];
    strcpy(data, "0123456789");
    // ERROR: src and dst overlap in a way that memcpy doesn't support.
    memcpy(data + 2, data, 10); 
}

void run() {
    std::cout << "\n--- Running Memory Abuser ---" << std::endl;
    trigger_use_after_free();
    trigger_heap_overflow();
    trigger_stack_overflow();
    trigger_uninitialized_read();
    trigger_memory_leak();
    trigger_mismatched_free();
    // trigger_double_free(); // Often crashes the program immediately, run if desired
    trigger_overlapping_memcpy();
}
} // namespace MemoryAbuser


// ====================================================================
// PART 2: THREADING NIGHTMARE - Data races & deadlocks for Helgrind/DRD
// ====================================================================
namespace ThreadingNightmare {

void racy_thread_func() {
    for (int i = 0; i < 200000; ++i) {
        // ERROR: Data race. No lock protecting the shared data.
        g_shared_racy_data++;
    }
}

void deadlock_thread_a() {
    std::cout << "[Thread] Deadlock thread A starting..." << std::endl;
    std::lock_guard<std::mutex> lock_a(g_mutex_a);
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    std::cout << "[Thread] A trying to get lock B..." << std::endl;
    std::lock_guard<std::mutex> lock_b(g_mutex_b); // Will likely deadlock here
    std::cout << "[Thread] A acquired both locks (should not happen)." << std::endl;
}

void deadlock_thread_b() {
    std::cout << "[Thread] Deadlock thread B starting..." << std::endl;
    std::lock_guard<std::mutex> lock_b(g_mutex_b);
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    std::cout << "[Thread] B trying to get lock A..." << std::endl;
    std::lock_guard<std::mutex> lock_a(g_mutex_a); // Will likely deadlock here
    std::cout << "[Thread] B acquired both locks (should not happen)." << std::endl;
}

namespace ProducerConsumerChaos {
    std::queue<int> g_queue;
    std::mutex g_queue_mutex;
    std::condition_variable g_cv;
    bool g_done = false;

    void producer() {
        for (int i = 0; i < 500; ++i) {
            {
                std::lock_guard<std::mutex> lock(g_queue_mutex);
                g_queue.push(i);
            }
            // ERROR: Notifying outside the lock can be fine, but the real bug
            // is in the consumer, which Helgrind can help diagnose.
            g_cv.notify_one();
        }
        {
            std::lock_guard<std::mutex> lock(g_queue_mutex);
            g_done = true;
        }
        g_cv.notify_all();
    }

    void consumer() {
        while (true) {
            std::unique_lock<std::mutex> lock(g_queue_mutex);
            g_cv.wait(lock, []{ return !g_queue.empty() || g_done; });

            if (g_done && g_queue.empty()) {
                break;
            }

            // BUG: We might wake up and find the queue empty if we don't re-check after wait.
            // A correct implementation uses a while loop: `while (!g_queue.empty())`.
            // Helgrind may detect potential race conditions around this pattern.
            if (!g_queue.empty()) {
                g_queue.pop();
            }
        }
    }
}


void run_data_race() {
    g_shared_racy_data = 0;
    std::cout << "[Thread] Spawning threads for data race..." << std::endl;
    std::vector<std::thread> threads;
    for (int i = 0; i < 4; ++i) threads.emplace_back(racy_thread_func);
    for (auto& t : threads) t.join();
    std::cout << "[Thread] Final racy value (expected " << 200000 * 4 << "): " << g_shared_racy_data << std::endl;
}

void run_deadlock() {
    std::cout << "[Thread] Spawning threads for deadlock (may hang)..." << std::endl;
    std::vector<std::thread> threads;
    threads.emplace_back(deadlock_thread_a);
    threads.emplace_back(deadlock_thread_b);
    // We expect a deadlock, so we won't join them in the main timed loop
    for (auto& t : threads) t.detach(); 
    std::cout << "[Thread] Deadlock threads detached. The program will not exit cleanly." << std::endl;
}

void run_producer_consumer() {
    std::cout << "[Thread] Running producer-consumer chaos..." << std::endl;
    ProducerConsumerChaos::g_done = false;
    std::vector<std::thread> threads;
    threads.emplace_back(ProducerConsumerChaos::producer);
    threads.emplace_back(ProducerConsumerChaos::consumer);
    threads.emplace_back(ProducerConsumerChaos::consumer);
    for(auto& t : threads) t.join();
}

void run() {
    std::cout << "\n--- Running Threading Nightmare ---" << std::endl;
    run_data_race();
    run_producer_consumer();
    // Note: run_deadlock() is commented out from the main timed loop because
    // it will hang the program, preventing other tests from cycling.
    // Run it manually if you wish to specifically test deadlock detection.
    // run_deadlock(); 
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
            strcpy((char*)mem, "mmap successful in child");
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
// PART 4: ALGORITHM STRESS TEST (for PGO/Callgrind)
// ====================================================================
namespace AlgorithmStress {

// --- 4a: Graph algorithm (Dijkstra's) ---
void run_dijkstra() {
    std::cout << "[Algo] Running Dijkstra's Algorithm..." << std::endl;
    int num_vertices = 500;
    using Edge = std::pair<int, int>;
    std::vector<std::list<Edge>> adj(num_vertices);

    // Create a dense graph to make it work harder
    for (int i = 0; i < num_vertices; ++i) {
        for (int j = 0; j < 10; ++j) {
            adj[i].push_back({(i * j) % 100 + 1, (i + j + 1) % num_vertices});
        }
    }

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
            if (dist[u] != std::numeric_limits<int>::max() && dist[u] + weight < dist[v]) {
                dist[v] = dist[u] + weight;
                pq.push({dist[v], v});
            }
        }
    }
}

// --- 4b: Recursive Backtracking (N-Queens) ---
namespace NQueens {
    int solve(int row, std::vector<int>& board) {
        int n = board.size();
        if (row == n) return 1;
        int count = 0;
        for (int col = 0; col < n; ++col) {
            bool is_safe = true;
            for (int prev_row = 0; prev_row < row; ++prev_row) {
                int prev_col = board[prev_row];
                if (prev_col == col || std::abs(prev_row - row) == std::abs(prev_col - col)) {
                    is_safe = false;
                    break;
                }
            }
            if (is_safe) {
                board[row] = col;
                count += solve(row + 1, board);
            }
        }
        return count;
    }
}
void run_n_queens() {
    int n = 12; // N=12 is substantially more work than N=9
    std::cout << "[Algo] Running N-Queens Solver (N="<< n << ")..." << std::endl;
    std::vector<int> board(n);
    NQueens::solve(0, board); // Stresses call stack
}

// --- 4c: Numerical Computation (Matrix Multiplication) ---
void run_matrix_multiplication() {
    std::cout << "[Algo] Running Matrix Multiplication..." << std::endl;
    int size = 250;
    std::vector<std::vector<int>> a(size, std::vector<int>(size));
    std::vector<std::vector<int>> b(size, std::vector<int>(size));
    std::vector<std::vector<int>> result(size, std::vector<int>(size, 0));

    for(int i=0; i<size; ++i) for(int j=0; j<size; ++j) { a[i][j] = i + j; b[i][j] = i - j; }

    // Triple nested loop is a classic hot spot for PGO/Callgrind to optimize
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
    run_matrix_multiplication();
}
} // namespace AlgorithmStress

// ====================================================================
// PART 5: ADVANCED PROFILING TARGETS (for Massif/Cachegrind)
// ====================================================================
namespace ProfilingTargets {

// --- 5a: Heap profiler target (Massif) ---
void run_massif_sawtooth() {
    std::cout << "[Profile] Running Massif 'sawtooth' heap stress..." << std::endl;
    std::vector<int*> memory_chunks;
    
    // 5 peaks of memory allocation
    for (int i = 0; i < 5; ++i) {
        // Allocate 1000 chunks of 10KB each (total ~10MB)
        for (int j = 0; j < 1000; ++j) {
            memory_chunks.push_back(new int[2500]); // 10000 bytes
        }
        // Deallocate all chunks
        for (int* chunk : memory_chunks) {
            delete[] chunk;
        }
        memory_chunks.clear();
    }
}

// --- 5b: Cache profiler target (Cachegrind) ---
namespace CacheStress {
    const int SIZE = 1024;
    std::vector<std::vector<int>> matrix(SIZE, std::vector<int>(SIZE));

    void transpose_naive() {
        // This is cache-unfriendly. When writing to `matrix[j][i]`, you jump
        // across huge memory regions, causing many cache misses.
        for (int i = 0; i < SIZE; ++i) {
            for (int j = 0; j < SIZE; ++j) {
                // Swap operation, but the access pattern is the key
                std::swap(matrix[i][j], matrix[j][i]);
            }
        }
    }
    
    void transpose_tiled(int blocksize) {
        // This is cache-friendly. It processes small blocks that fit in cache.
        for (int i = 0; i < SIZE; i += blocksize) {
            for (int j = 0; j < SIZE; j += blocksize) {
                // Transpose the block
                for (int k = i; k < i + blocksize; ++k) {
                    for (int l = j; l < j + blocksize; ++l) {
                        std::swap(matrix[k][l], matrix[l][k]);
                    }
                }
            }
        }
    }
}

void run_cache_stress() {
    std::cout << "[Profile] Running Cachegrind stress (naive vs. tiled)..." << std::endl;
    // Initializing matrix data once
    for(int i=0; i<CacheStress::SIZE; ++i) for(int j=0; j<CacheStress::SIZE; ++j) CacheStress::matrix[i][j] = i*j;

    // Run the inefficient version
    CacheStress::transpose_naive();
    
    // Run the efficient version. A profiler like Cachegrind will show
    // a dramatic reduction in cache misses (D1 misses, L3 misses).
    CacheStress::transpose_tiled(32);
}

void run() {
    std::cout << "\n--- Running Advanced Profiling Targets ---" << std::endl;
    run_massif_sawtooth();
    run_cache_stress();
}

} // namespace ProfilingTargets


// ====================================================================
// --- MAIN ORCHESTRATOR ---
// ====================================================================
void run_all_tests() {
    MemoryAbuser::run();
    ThreadingNightmare::run();
    SyscallChaos::run();
    AlgorithmStress::run();
    ProfilingTargets::run();
}

int main(int argc, char* argv[]) {
    const long long RUN_DURATION_SECONDS = 300; // 5 minutes
    std::cout << "Starting ultimate Valgrind stress test..." << std::endl;
    std::cout << "Target duration: " << RUN_DURATION_SECONDS << " seconds." << std::endl;

    std::map<std::string, std::function<void()>> test_suites = {
        {"memory", MemoryAbuser::run},
        {"threads", ThreadingNightmare::run},
        {"syscall", SyscallChaos::run},
        {"algo", AlgorithmStress::run},
        {"profile", ProfilingTargets::run},
        {"all", run_all_tests}
    };

    if (argc > 1) {
        std::string arg = argv[1];
        if (test_suites.count(arg)) {
            std::cout << "Running only the '" << arg << "' test suite." << std::endl;
            test_suites[arg]();
        } else {
            std::cerr << "Unknown test suite: " << arg << std::endl;
            std::cerr << "Available suites: memory, threads, syscall, algo, profile, all" << std::endl;
            return 1;
        }
    } else {
        std::cout << "Running all test suites in a loop." << std::endl;
        auto start_time = std::chrono::steady_clock::now();
        int cycle = 0;
        while (true) {
            auto current_time = std::chrono::steady_clock::now();
            long long elapsed_seconds = std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time).count();
            
            if (elapsed_seconds >= RUN_DURATION_SECONDS) {
                break;
            }

            std::cout << "\n=========================================" << std::endl;
            std::cout << "Starting Test Cycle " << ++cycle << " (Elapsed: " << elapsed_seconds << "s)" << std::endl;
            std::cout << "=========================================\n" << std::endl;
            
            run_all_tests();
        }
    }

    std::cout << "\nUltimate stress test finished." << std::endl;
    return 0;
}
