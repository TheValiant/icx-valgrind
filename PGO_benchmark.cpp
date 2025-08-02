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
#include <random>
#include <set>
#include <unordered_map>
#include <memory>
#include <future>
#include <bitset>
#include <complex>
#include <cmath>

// System headers
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <signal.h>
#include <sys/resource.h> // For setrlimit to disable core dumps

// --- Global state for threading nightmares ---
std::mutex g_mutex_a;
std::mutex g_mutex_b;
long g_shared_racy_data = 0;
std::atomic<bool> g_should_exit{false};
std::mt19937 g_rng(std::chrono::steady_clock::now().time_since_epoch().count());

// Enhanced test control and timing
struct TestStats {
    std::string name;
    std::chrono::steady_clock::time_point start_time;
    std::chrono::steady_clock::time_point end_time;
    bool completed;
    
    TestStats(const std::string& n) : name(n), completed(false) {}
    
    void start() {
        start_time = std::chrono::steady_clock::now();
        std::cout << "[TIMING] Starting test: " << name << std::endl;
    }
    
    void finish() {
        end_time = std::chrono::steady_clock::now();
        completed = true;
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        std::cout << "[TIMING] Completed test: " << name << " in " << duration << "ms" << std::endl;
    }
};

std::vector<TestStats> g_test_history;

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
    // Use a very controlled overflow that Valgrind can catch
    // Initialize buffer first to prevent uninitialized access warnings
    memset(buffer, 0, sizeof(buffer));
    
    // Small, controlled overflow that's less likely to crash
    for (int i = 0; i < 20; ++i) {
        if (i >= 16) {
            // This will trigger stack overflow detection in Valgrind
            // but is less likely to crash immediately
            buffer[i] = 'A' + (i % 26);
        } else {
            buffer[i] = 'a' + (i % 26);
        }
    }
    
    // Make sure the data is "used" to prevent compiler optimization
    volatile char temp = buffer[18];
    (void)temp;
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
    // This is very likely to crash, so we'll comment it out for now
    // and let Valgrind detect the other errors instead
    // delete p;
    std::cout << "[Mem] Double free skipped to prevent crash." << std::endl;
}

void trigger_overlapping_memcpy() {
    std::cout << "[Mem] Triggering overlapping src/dst in memcpy..." << std::endl;
    char data[20];
    strcpy(data, "0123456789");
    // ERROR: src and dst overlap in a way that memcpy doesn't support.
    // Use memmove instead of memcpy for actual safety, but memcpy for Valgrind detection
    // Reduce overlap to be less likely to crash
    memcpy(data + 1, data, 8); // Smaller overlap
}

// Enhanced Memcheck tests for subtle errors
void trigger_off_by_one_errors() {
    std::cout << "[Mem] Triggering off-by-one errors..." << std::endl;
    
    // Stack buffer off-by-one
    char stack_buffer[10];
    for (int i = 0; i <= 10; ++i) { // ERROR: i should be < 10
        stack_buffer[i] = 'A' + (i % 26);
    }
    
    // Heap buffer off-by-one
    int* heap_buffer = new int[5];
    for (int i = 0; i <= 5; ++i) { // ERROR: i should be < 5
        heap_buffer[i] = i * i;
    }
    delete[] heap_buffer;
    
    // String off-by-one
    char* str = new char[10];
    strncpy(str, "0123456789", 10); // ERROR: no null terminator space
    str[9] = '\0'; // This overwrites the last valid character
    delete[] str;
}

void trigger_partial_overlaps() {
    std::cout << "[Mem] Triggering partial memory overlaps..." << std::endl;
    
    char buffer[100];
    strcpy(buffer, "This is a test string for overlap detection");
    
    // Partial overlap in memmove (this is actually safe, but tests the tool)
    memmove(buffer + 5, buffer, 20);
    
    // Partial overlap in memcpy (this is unsafe)
    char buffer2[50];
    strcpy(buffer2, "Another test string");
    memcpy(buffer2 + 3, buffer2, 15); // ERROR: overlapping regions
}

void trigger_complex_allocation_patterns() {
    std::cout << "[Mem] Triggering complex allocation/deallocation patterns..." << std::endl;
    
    std::vector<void*> ptrs;
    
    // Pattern 1: Alternating allocation and deallocation
    for (int i = 0; i < 20; ++i) {
        ptrs.push_back(malloc(100 + i * 10));
        if (i > 5 && i % 3 == 0) {
            free(ptrs[i - 3]);
            ptrs[i - 3] = nullptr;
        }
    }
    
    // Pattern 2: Mixed new/malloc (potential for mismatched free)
    void* malloced = malloc(200);
    int* newed = new int[50];
    
    // ERROR: Wrong deallocation methods
    // free(newed);  // This would be an error - commented out to prevent crash
    // delete malloced;  // This would be an error - commented out to prevent crash
    
    // Correct cleanup
    free(malloced);
    delete[] newed;
    
    // Clean up remaining allocations
    for (void* ptr : ptrs) {
        if (ptr) free(ptr);
    }
}

void trigger_memory_alignment_issues() {
    std::cout << "[Mem] Triggering memory alignment issues..." << std::endl;
    
    // Allocate buffer and create misaligned access
    char* buffer = new char[100];
    
    // Force misaligned access to larger types
    char* misaligned_ptr = buffer + 1; // Not aligned for int/double
    
    // These accesses may trigger alignment warnings in Valgrind
    *(int*)misaligned_ptr = 0x12345678;
    *(double*)(misaligned_ptr + 4) = 3.14159;
    
    // Read back the misaligned data
    volatile int misaligned_int = *(int*)misaligned_ptr;
    volatile double misaligned_double = *(double*)(misaligned_ptr + 4);
    
    (void)misaligned_int; (void)misaligned_double; // Prevent optimization
    
    delete[] buffer;
    
    // Test with different alignment patterns
    for (int offset = 1; offset < 8; ++offset) {
        char* test_buffer = new char[64];
        char* offset_ptr = test_buffer + offset;
        
        // Try to access as 8-byte aligned data
        if (offset_ptr + sizeof(long long) < test_buffer + 64) {
            *(long long*)offset_ptr = 0xDEADBEEFCAFEBABE;
            volatile long long val = *(long long*)offset_ptr;
            (void)val;
        }
        
        delete[] test_buffer;
    }
}

void run() {
    TestStats test_stats("MemoryAbuser");
    test_stats.start();
    
    std::cout << "\n--- Running Enhanced Memory Abuser ---" << std::endl;
    
    // Enhanced test selection - ensure all critical tests run
    std::vector<std::pair<std::string, std::function<void()>>> memory_tests = {
        {"use_after_free", []() {
            try { trigger_use_after_free(); }
            catch (...) { std::cout << "[Mem] Use-after-free caught exception." << std::endl; }
        }},
        {"heap_overflow", []() {
            try { trigger_heap_overflow(); }
            catch (...) { std::cout << "[Mem] Heap overflow caught exception." << std::endl; }
        }},
        {"stack_overflow", []() {
            try { trigger_stack_overflow(); }
            catch (...) { std::cout << "[Mem] Stack overflow caught exception." << std::endl; }
        }},
        {"uninitialized_read", []() {
            try { trigger_uninitialized_read(); }
            catch (...) { std::cout << "[Mem] Uninitialized read caught exception." << std::endl; }
        }},
        {"memory_leak", []() {
            try { trigger_memory_leak(); }
            catch (...) { std::cout << "[Mem] Memory leak caught exception." << std::endl; }
        }},
        {"mismatched_free", []() {
            try { trigger_mismatched_free(); }
            catch (...) { std::cout << "[Mem] Mismatched free caught exception." << std::endl; }
        }},
        {"double_free", []() {
            try { trigger_double_free(); }
            catch (...) { std::cout << "[Mem] Double free caught exception." << std::endl; }
        }},
        {"overlapping_memcpy", []() {
            try { trigger_overlapping_memcpy(); }
            catch (...) { std::cout << "[Mem] Overlapping memcpy caught exception." << std::endl; }
        }},
        {"off_by_one", []() {
            try { trigger_off_by_one_errors(); }
            catch (...) { std::cout << "[Mem] Off-by-one errors caught exception." << std::endl; }
        }},
        {"partial_overlaps", []() {
            try { trigger_partial_overlaps(); }
            catch (...) { std::cout << "[Mem] Partial overlaps caught exception." << std::endl; }
        }},
        {"complex_allocation", []() {
            try { trigger_complex_allocation_patterns(); }
            catch (...) { std::cout << "[Mem] Complex allocation caught exception." << std::endl; }
        }},
        {"alignment_issues", []() {
            try { trigger_memory_alignment_issues(); }
            catch (...) { std::cout << "[Mem] Alignment issues caught exception." << std::endl; }
        }}
    };
    
    // Shuffle for randomization but ensure critical tests always run
    std::shuffle(memory_tests.begin(), memory_tests.end(), g_rng);
    
    // Always run the first 8 tests (core functionality)
    int core_tests = std::min(8, (int)memory_tests.size());
    for (int i = 0; i < core_tests; ++i) {
        std::cout << "[Mem] Running " << memory_tests[i].first << "..." << std::endl;
        auto test_start = std::chrono::steady_clock::now();
        memory_tests[i].second();
        auto test_end = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(test_end - test_start).count();
        std::cout << "[Mem] " << memory_tests[i].first << " completed in " << duration << "ms" << std::endl;
    }
    
    // Run additional tests based on available time
    std::uniform_int_distribution<int> extra_dist(0, memory_tests.size() - core_tests);
    int extra_tests = extra_dist(g_rng);
    for (int i = core_tests; i < core_tests + extra_tests && i < memory_tests.size(); ++i) {
        std::cout << "[Mem] Running additional " << memory_tests[i].first << "..." << std::endl;
        memory_tests[i].second();
    }
    
    test_stats.finish();
    g_test_history.push_back(test_stats);
}
} // namespace MemoryAbuser


// ====================================================================
// PART 2: THREADING NIGHTMARE - Data races & deadlocks for Helgrind/DRD
// ====================================================================
namespace ThreadingNightmare {

void racy_thread_func() {
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < 200000 && !g_should_exit.load(); ++i) {
        // ERROR: Data race. No lock protecting the shared data.
        g_shared_racy_data++;
        
        // Add timeout check every 10000 iterations
        if (i % 10000 == 0) {
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now() - start).count();
            if (elapsed > 10) { // 10 second timeout per thread
                break;
            }
        }
    }
}

void deadlock_thread_a() {
    if (g_should_exit.load()) return;
    std::cout << "[Thread] Deadlock thread A starting..." << std::endl;
    
    auto start = std::chrono::steady_clock::now();
    if (g_mutex_a.try_lock()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        std::cout << "[Thread] A trying to get lock B..." << std::endl;
        
        // Check for timeout
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - start).count();
        if (elapsed > 5 || g_should_exit.load()) {
            std::cout << "[Thread] A timed out." << std::endl;
            g_mutex_a.unlock();
            return;
        }
        
        if (g_mutex_b.try_lock()) {
            std::cout << "[Thread] A acquired both locks." << std::endl;
            g_mutex_b.unlock();
        } else {
            std::cout << "[Thread] A couldn't get lock B." << std::endl;
        }
        g_mutex_a.unlock();
    } else {
        std::cout << "[Thread] A couldn't get lock A." << std::endl;
    }
}

void deadlock_thread_b() {
    if (g_should_exit.load()) return;
    std::cout << "[Thread] Deadlock thread B starting..." << std::endl;
    
    auto start = std::chrono::steady_clock::now();
    if (g_mutex_b.try_lock()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        std::cout << "[Thread] B trying to get lock A..." << std::endl;
        
        // Check for timeout
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - start).count();
        if (elapsed > 5 || g_should_exit.load()) {
            std::cout << "[Thread] B timed out." << std::endl;
            g_mutex_b.unlock();
            return;
        }
        
        if (g_mutex_a.try_lock()) {
            std::cout << "[Thread] B acquired both locks." << std::endl;
            g_mutex_a.unlock();
        } else {
            std::cout << "[Thread] B couldn't get lock A." << std::endl;
        }
        g_mutex_b.unlock();
    } else {
        std::cout << "[Thread] B couldn't get lock B." << std::endl;
    }
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
    std::cout << "[Thread] Spawning threads for deadlock prevention test..." << std::endl;
    std::vector<std::thread> threads;
    threads.emplace_back(deadlock_thread_a);
    threads.emplace_back(deadlock_thread_b);
    
    // Use timeout to prevent hanging
    auto start = std::chrono::steady_clock::now();
    for (auto& t : threads) {
        if (t.joinable()) {
            t.join();
        }
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - start).count();
        if (elapsed > 10) { // 10 second timeout
            std::cout << "[Thread] Deadlock test timed out, continuing..." << std::endl;
            break;
        }
    }
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
    
    // Randomize the order of threading tests
    std::vector<std::function<void()>> thread_tests = {
        run_data_race,
        run_producer_consumer,
        run_deadlock
    };
    
    std::shuffle(thread_tests.begin(), thread_tests.end(), g_rng);
    
    // Run a random subset
    std::uniform_int_distribution<int> count_dist(1, thread_tests.size());
    int num_to_run = count_dist(g_rng);
    
    for (int i = 0; i < num_to_run; ++i) {
        if (g_should_exit.load()) break;
        thread_tests[i]();
    }
}

} // namespace ThreadingNightmare


// ====================================================================
// PART 3: IPC & SYSCALL CHAOS - fork, pipes, fds, mmap
// ====================================================================
namespace SyscallChaos {

void run_crash_signal_test() {
    std::cout << "[Syscall] Fork and random crash signal test..." << std::endl;
    
    pid_t pid = fork();
    if (pid == -1) { 
        perror("fork"); 
        return; 
    }

    if (pid == 0) { // Child process
        std::cout << "[Child] Child process started, doing random work..." << std::endl;
        
        // Child does some random computational work
        std::uniform_int_distribution<int> work_dist(1000, 5000);
        int work_amount = work_dist(g_rng);
        
        volatile long sum = 0;
        for (int i = 0; i < work_amount; ++i) {
            sum += i * i + (i % 7) * (i % 11);
            
            // Small random delays to make the work take some time
            if (i % 100 == 0) {
                std::this_thread::sleep_for(std::chrono::microseconds(10));
            }
        }
        
        std::cout << "[Child] Child completed work, sum = " << sum << std::endl;
        _exit(0);
    } else { // Parent process
        // Give child a moment to start its work
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        
        // Send a random crash signal to the child
        std::vector<int> crash_signals = {SIGTERM, SIGKILL, SIGINT, SIGUSR1, SIGUSR2};
        std::uniform_int_distribution<size_t> signal_dist(0, crash_signals.size() - 1);
        int chosen_signal = crash_signals[signal_dist(g_rng)];
        
        std::cout << "[Parent] Sending signal " << chosen_signal << " to child process " << pid << std::endl;
        
        if (kill(pid, chosen_signal) == -1) {
            perror("kill");
        }
        
        // Wait for child and get its exit status
        int status;
        pid_t result = waitpid(pid, &status, 0);
        if (result != -1) {
            if (WIFEXITED(status)) {
                std::cout << "[Parent] Child exited normally with code " << WEXITSTATUS(status) << std::endl;
            } else if (WIFSIGNALED(status)) {
                std::cout << "[Parent] Child terminated by signal " << WTERMSIG(status) << std::endl;
            }
        } else {
            perror("waitpid");
        }
    }
}

void run() {
    std::cout << "\n--- Running IPC and Syscall Chaos ---" << std::endl;
    
    // Randomize the order of syscall tests
    std::vector<std::function<void()>> syscall_tests = {
        []() {
            // Original pipe/mmap test
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
        },
        run_crash_signal_test,
        []() {
            std::cout << "[Syscall] Leaking a file descriptor..." << std::endl;
            // ERROR: This file descriptor will be leaked.
            (void)open("/dev/null", O_RDONLY);
        }
    };
    
    std::shuffle(syscall_tests.begin(), syscall_tests.end(), g_rng);
    
    // Run a random subset
    std::uniform_int_distribution<int> count_dist(1, syscall_tests.size());
    int num_to_run = count_dist(g_rng);
    
    for (int i = 0; i < num_to_run; ++i) {
        if (g_should_exit.load()) break;
        syscall_tests[i]();
    }
}
} // namespace SyscallChaos


// ====================================================================
// PART 4: ALGORITHM STRESS TEST (for PGO/Callgrind)
// ====================================================================
namespace AlgorithmStress {

// --- 4a: Graph algorithm (Dijkstra's) - Enhanced CPU intensity ---
void run_dijkstra() {
    std::cout << "[Algo] Running Enhanced Dijkstra's Algorithm..." << std::endl;
    int num_vertices = 1000; // Increased from 500
    using Edge = std::pair<int, int>;
    std::vector<std::list<Edge>> adj(num_vertices);

    // Create a much denser graph with more complex weight calculations
    for (int i = 0; i < num_vertices; ++i) {
        for (int j = 0; j < 25; ++j) { // Increased from 10
            // More complex weight calculation for CPU intensity
            int weight = (i * j * 17 + i * i + j * j) % 150 + 1;
            int target = (i + j * 7 + weight) % num_vertices;
            adj[i].push_back({weight, target});
        }
    }

    std::priority_queue<Edge, std::vector<Edge>, std::greater<Edge>> pq;
    std::vector<int> dist(num_vertices, std::numeric_limits<int>::max());

    // Run multiple iterations from different start nodes for better profiling
    std::vector<int> start_nodes = {0, num_vertices/4, num_vertices/2, 3*num_vertices/4};
    
    for (int start_node : start_nodes) {
        std::fill(dist.begin(), dist.end(), std::numeric_limits<int>::max());
        while (!pq.empty()) pq.pop(); // Clear priority queue
        
        pq.push({0, start_node});
        dist[start_node] = 0;
        int processed_nodes = 0;

        while (!pq.empty() && processed_nodes < num_vertices) {
            int u = pq.top().second;
            int d = pq.top().first;
            pq.pop();
            processed_nodes++;
            
            if (d > dist[u]) continue; // Skip outdated entries

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

// Forward declarations for new algorithms
void run_red_black_tree();
void run_fft();
void run_tsp();
void run_suffix_array();

void run() {
    std::cout << "\n--- Running Advanced Algorithm Stress Test ---" << std::endl;
    
    // Randomize the order of algorithm execution
    std::vector<std::function<void()>> algorithms = {
        run_dijkstra,
        run_n_queens,
        run_matrix_multiplication,
        run_red_black_tree,
        run_fft,
        run_tsp,
        run_suffix_array
    };
    
    std::shuffle(algorithms.begin(), algorithms.end(), g_rng);
    
    // Run a random subset of algorithms
    std::uniform_int_distribution<int> count_dist(3, algorithms.size());
    int num_to_run = count_dist(g_rng);
    
    for (int i = 0; i < num_to_run; ++i) {
        if (g_should_exit.load()) break;
        algorithms[i]();
    }
}

// --- 4d: Red-Black Tree Implementation ---
namespace RedBlackTree {
    enum Color { RED, BLACK };
    
    struct Node {
        int data;
        Color color;
        std::shared_ptr<Node> left, right, parent;
        Node(int val) : data(val), color(RED), left(nullptr), right(nullptr), parent(nullptr) {}
    };
    
    class RBTree {
        std::shared_ptr<Node> root;
        
        void rotateLeft(std::shared_ptr<Node> x) {
            auto y = x->right;
            x->right = y->left;
            if (y->left) y->left->parent = x;
            y->parent = x->parent;
            if (!x->parent) root = y;
            else if (x == x->parent->left) x->parent->left = y;
            else x->parent->right = y;
            y->left = x;
            x->parent = y;
        }
        
        void rotateRight(std::shared_ptr<Node> x) {
            auto y = x->left;
            x->left = y->right;
            if (y->right) y->right->parent = x;
            y->parent = x->parent;
            if (!x->parent) root = y;
            else if (x == x->parent->right) x->parent->right = y;
            else x->parent->left = y;
            y->right = x;
            x->parent = y;
        }
        
        void fixInsert(std::shared_ptr<Node> z) {
            while (z->parent && z->parent->color == RED) {
                if (z->parent == z->parent->parent->left) {
                    auto y = z->parent->parent->right;
                    if (y && y->color == RED) {
                        z->parent->color = BLACK;
                        y->color = BLACK;
                        z->parent->parent->color = RED;
                        z = z->parent->parent;
                    } else {
                        if (z == z->parent->right) {
                            z = z->parent;
                            rotateLeft(z);
                        }
                        z->parent->color = BLACK;
                        z->parent->parent->color = RED;
                        rotateRight(z->parent->parent);
                    }
                } else {
                    auto y = z->parent->parent->left;
                    if (y && y->color == RED) {
                        z->parent->color = BLACK;
                        y->color = BLACK;
                        z->parent->parent->color = RED;
                        z = z->parent->parent;
                    } else {
                        if (z == z->parent->left) {
                            z = z->parent;
                            rotateRight(z);
                        }
                        z->parent->color = BLACK;
                        z->parent->parent->color = RED;
                        rotateLeft(z->parent->parent);
                    }
                }
            }
            root->color = BLACK;
        }
        
    public:
        void insert(int data) {
            auto z = std::make_shared<Node>(data);
            auto y = std::shared_ptr<Node>(nullptr);
            auto x = root;
            
            while (x) {
                y = x;
                if (z->data < x->data) x = x->left;
                else x = x->right;
            }
            
            z->parent = y;
            if (!y) root = z;
            else if (z->data < y->data) y->left = z;
            else y->right = z;
            
            if (!z->parent) {
                z->color = BLACK;
                return;
            }
            if (!z->parent->parent) return;
            
            fixInsert(z);
        }
    };
}

void run_red_black_tree() {
    std::cout << "[Algo] Running Red-Black Tree operations..." << std::endl;
    RedBlackTree::RBTree tree;
    std::uniform_int_distribution<int> dist(1, 10000);
    for (int i = 0; i < 1000; ++i) {
        tree.insert(dist(g_rng));
    }
}

// --- 4e: Fast Fourier Transform ---
namespace FFT {
    using Complex = std::complex<double>;
    
    void fft(std::vector<Complex>& a) {
        int n = a.size();
        if (n <= 1) return;
        
        std::vector<Complex> even(n/2), odd(n/2);
        for (int i = 0; i < n/2; ++i) {
            even[i] = a[2*i];
            odd[i] = a[2*i + 1];
        }
        
        fft(even);
        fft(odd);
        
        for (int i = 0; i < n/2; ++i) {
            Complex t = std::polar(1.0, -2 * M_PI * i / n) * odd[i];
            a[i] = even[i] + t;
            a[i + n/2] = even[i] - t;
        }
    }
}

void run_fft() {
    std::cout << "[Algo] Running Fast Fourier Transform..." << std::endl;
    std::vector<std::complex<double>> signal(1024);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    
    for (auto& sample : signal) {
        sample = std::complex<double>(dist(g_rng), 0);
    }
    
    FFT::fft(signal);
}

// --- 4f: Traveling Salesman Problem (Dynamic Programming) ---
void run_tsp() {
    std::cout << "[Algo] Running TSP with Dynamic Programming..." << std::endl;
    const int n = 15; // Small enough to complete in reasonable time
    std::vector<std::vector<int>> dist(n, std::vector<int>(n));
    
    // Generate random distance matrix
    std::uniform_int_distribution<int> dist_gen(1, 100);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j) dist[i][j] = 0;
            else dist[i][j] = dist_gen(g_rng);
        }
    }
    
    // DP solution for TSP
    std::vector<std::vector<int>> dp(1 << n, std::vector<int>(n, std::numeric_limits<int>::max()));
    dp[1][0] = 0;
    
    for (int mask = 0; mask < (1 << n); ++mask) {
        for (int u = 0; u < n; ++u) {
            if (!(mask & (1 << u)) || dp[mask][u] == std::numeric_limits<int>::max()) continue;
            for (int v = 0; v < n; ++v) {
                if (mask & (1 << v)) continue;
                int new_mask = mask | (1 << v);
                dp[new_mask][v] = std::min(dp[new_mask][v], dp[mask][u] + dist[u][v]);
            }
        }
    }
}

// --- 4g: Suffix Array Construction ---
void run_suffix_array() {
    std::cout << "[Algo] Running Suffix Array construction..." << std::endl;
    std::string text = "banana$";
    for (int i = 0; i < 100; ++i) {
        text += "abracadabra" + std::to_string(i);
    }
    text += "$";
    
    int n = text.length();
    std::vector<int> sa(n), rank(n), tmp(n);
    
    // Initial ranking
    for (int i = 0; i < n; ++i) {
        sa[i] = i;
        rank[i] = text[i];
    }
    
    for (int k = 1; k < n; k <<= 1) {
        auto cmp = [&](int i, int j) {
            if (rank[i] != rank[j]) return rank[i] < rank[j];
            int ri = (i + k < n) ? rank[i + k] : -1;
            int rj = (j + k < n) ? rank[j + k] : -1;
            return ri < rj;
        };
        
        std::sort(sa.begin(), sa.end(), cmp);
        
        tmp[sa[0]] = 0;
        for (int i = 1; i < n; ++i) {
            tmp[sa[i]] = tmp[sa[i-1]] + (cmp(sa[i-1], sa[i]) ? 1 : 0);
        }
        rank = tmp;
    }
}
} // namespace AlgorithmStress

// ====================================================================
// PART 5: ADVANCED PROFILING TARGETS (for Massif/Cachegrind)
// ====================================================================
namespace ProfilingTargets {

// --- 5a: Enhanced heap profiler target (Massif) ---
void run_massif_sawtooth() {
    std::cout << "[Profile] Running Enhanced Massif heap stress with diverse patterns..." << std::endl;
    
    // Pattern 1: Classic sawtooth
    std::vector<int*> memory_chunks;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 500; ++j) {
            memory_chunks.push_back(new int[2500]); // 10KB chunks
        }
        for (int* chunk : memory_chunks) {
            delete[] chunk;
        }
        memory_chunks.clear();
    }
    
    // Pattern 2: Fragmentation pattern - allocate different sizes
    std::vector<char*> mixed_chunks;
    std::uniform_int_distribution<int> size_dist(100, 5000);
    for (int i = 0; i < 1000; ++i) {
        int size = size_dist(g_rng);
        mixed_chunks.push_back(new char[size]);
        
        // Randomly deallocate some chunks to create fragmentation
        if (i > 100 && g_rng() % 3 == 0) {
            int idx = g_rng() % mixed_chunks.size();
            if (mixed_chunks[idx]) {
                delete[] mixed_chunks[idx];
                mixed_chunks[idx] = nullptr;
            }
        }
    }
    
    // Clean up remaining chunks
    for (char* chunk : mixed_chunks) {
        if (chunk) delete[] chunk;
    }
    mixed_chunks.clear();
    
    // Pattern 3: Large allocation bursts
    std::vector<double*> large_chunks;
    for (int burst = 0; burst < 5; ++burst) {
        for (int i = 0; i < 50; ++i) {
            large_chunks.push_back(new double[50000]); // 400KB chunks
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    
    // Deallocate in reverse order for different deallocation pattern
    for (auto it = large_chunks.rbegin(); it != large_chunks.rend(); ++it) {
        delete[] *it;
    }
    
    // Pattern 4: Persistent allocations with gradual growth
    static std::vector<std::vector<long>*> persistent_data;
    for (int i = 0; i < 20; ++i) {
        auto* vec = new std::vector<long>(1000 + i * 500);
        persistent_data.push_back(vec);
    }
}

// --- 5b: Enhanced cache profiler target (Cachegrind) ---
namespace CacheStress {
    const int SIZE = 1024;
    const int LARGE_SIZE = 2048;
    std::vector<std::vector<int>> matrix(SIZE, std::vector<int>(SIZE));
    std::vector<int> large_array(LARGE_SIZE * LARGE_SIZE);
    std::vector<int*> pointer_chase_array;

    void transpose_naive() {
        // Cache-unfriendly matrix transpose
        for (int i = 0; i < SIZE; ++i) {
            for (int j = 0; j < SIZE; ++j) {
                std::swap(matrix[i][j], matrix[j][i]);
            }
        }
    }
    
    void transpose_tiled(int blocksize) {
        // Cache-friendly tiled transpose
        for (int i = 0; i < SIZE; i += blocksize) {
            for (int j = 0; j < SIZE; j += blocksize) {
                for (int k = i; k < i + blocksize; ++k) {
                    for (int l = j; l < j + blocksize; ++l) {
                        std::swap(matrix[k][l], matrix[l][k]);
                    }
                }
            }
        }
    }
    
    void random_access_pattern() {
        // Extremely cache-unfriendly random access
        std::uniform_int_distribution<int> dist(0, LARGE_SIZE * LARGE_SIZE - 1);
        for (int i = 0; i < 100000; ++i) {
            int idx = dist(g_rng);
            large_array[idx] += i % 255;
        }
    }
    
    void pointer_chasing() {
        // Set up pointer chasing pattern - very cache unfriendly
        const int CHASE_SIZE = 10000;
        pointer_chase_array.resize(CHASE_SIZE);
        
        // Allocate scattered memory chunks
        for (int i = 0; i < CHASE_SIZE; ++i) {
            pointer_chase_array[i] = new int[100];
        }
        
        // Create random pointer chase pattern
        std::vector<int> indices(CHASE_SIZE);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), g_rng);
        
        // Follow the pointer chain
        int current = 0;
        for (int i = 0; i < 50000; ++i) {
            current = indices[current % CHASE_SIZE];
            pointer_chase_array[current][i % 100] += 1;
        }
        
        // Clean up
        for (int* ptr : pointer_chase_array) {
            delete[] ptr;
        }
        pointer_chase_array.clear();
    }
    
    void strided_access() {
        // Cache-unfriendly strided access pattern
        const int STRIDE = 64; // Skip cache lines
        for (int stride = 1; stride <= STRIDE; stride *= 2) {
            for (int i = 0; i < LARGE_SIZE * LARGE_SIZE; i += stride) {
                large_array[i] += stride;
            }
        }
    }
}

void run_cache_stress() {
    std::cout << "[Profile] Running Enhanced Cachegrind stress (multiple patterns)..." << std::endl;
    
    // Initialize data structures
    for(int i=0; i<CacheStress::SIZE; ++i) {
        for(int j=0; j<CacheStress::SIZE; ++j) {
            CacheStress::matrix[i][j] = i*j;
        }
    }
    
    for(int i=0; i<CacheStress::LARGE_SIZE * CacheStress::LARGE_SIZE; ++i) {
        CacheStress::large_array[i] = i % 1000;
    }

    // Run multiple cache stress patterns
    std::cout << "[Profile] Running naive transpose..." << std::endl;
    CacheStress::transpose_naive();
    
    std::cout << "[Profile] Running tiled transpose..." << std::endl;
    CacheStress::transpose_tiled(32);
    
    std::cout << "[Profile] Running random access pattern..." << std::endl;
    CacheStress::random_access_pattern();
    
    std::cout << "[Profile] Running pointer chasing..." << std::endl;
    CacheStress::pointer_chasing();
    
    std::cout << "[Profile] Running strided access..." << std::endl;
    CacheStress::strided_access();
}

void run() {
    std::cout << "\n--- Running Advanced Profiling Targets ---" << std::endl;
    run_massif_sawtooth();
    run_cache_stress();
}

} // namespace ProfilingTargets


// ====================================================================
// --- ENHANCED MAIN ORCHESTRATOR WITH PROGRESS REPORTING ---
// ====================================================================

void print_test_summary() {
    std::cout << "\n=== TEST EXECUTION SUMMARY ===" << std::endl;
    long long total_time = 0;
    int completed_tests = 0;
    
    for (const auto& test : g_test_history) {
        if (test.completed) {
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                test.end_time - test.start_time).count();
            total_time += duration;
            completed_tests++;
            std::cout << "✓ " << test.name << ": " << duration << "ms" << std::endl;
        } else {
            std::cout << "✗ " << test.name << ": INCOMPLETE" << std::endl;
        }
    }
    
    std::cout << "Total tests completed: " << completed_tests << "/" << g_test_history.size() << std::endl;
    std::cout << "Total execution time: " << total_time << "ms" << std::endl;
    std::cout << "================================" << std::endl;
}

void run_all_tests() {
    TestStats orchestrator_stats("TestOrchestrator");
    orchestrator_stats.start();
    
    std::cout << "[PROGRESS] Starting comprehensive test suite..." << std::endl;
    
    // Enhanced test selection with priority system
    struct TestInfo {
        std::string name;
        std::function<void()> func;
        int priority; // Higher = more important
        bool always_run;
    };
    
    std::vector<TestInfo> test_functions = {
        {"MemoryAbuser", MemoryAbuser::run, 10, true},
        {"ThreadingNightmare", ThreadingNightmare::run, 8, true}, 
        {"AlgorithmStress", AlgorithmStress::run, 7, false},
        {"ProfilingTargets", ProfilingTargets::run, 6, false},
        {"SyscallChaos", SyscallChaos::run, 5, false}
    };
    
    // Sort by priority (highest first)
    std::sort(test_functions.begin(), test_functions.end(), 
              [](const TestInfo& a, const TestInfo& b) { return a.priority > b.priority; });
    
    // Always run high-priority tests
    std::vector<TestInfo> selected_tests;
    for (const auto& test : test_functions) {
        if (test.always_run) {
            selected_tests.push_back(test);
        }
    }
    
    // Randomly select additional tests
    std::vector<TestInfo> optional_tests;
    for (const auto& test : test_functions) {
        if (!test.always_run) {
            optional_tests.push_back(test);
        }
    }
    
    std::shuffle(optional_tests.begin(), optional_tests.end(), g_rng);
    std::uniform_int_distribution<int> count_dist(1, optional_tests.size());
    int num_optional = count_dist(g_rng);
    
    for (int i = 0; i < num_optional; ++i) {
        selected_tests.push_back(optional_tests[i]);
    }
    
    // Execute selected tests with progress reporting
    int test_num = 0;
    for (const auto& test_info : selected_tests) {
        if (g_should_exit.load()) break;
        
        test_num++;
        std::cout << "\n[PROGRESS] Running test " << test_num << "/" << selected_tests.size() 
                  << ": " << test_info.name << std::endl;
        std::cout << "[PROGRESS] Priority: " << test_info.priority << std::endl;
        
        auto test_start = std::chrono::steady_clock::now();
        test_info.func();
        auto test_end = std::chrono::steady_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(test_end - test_start).count();
        std::cout << "[PROGRESS] Completed " << test_info.name << " in " << duration << "ms" << std::endl;
    }
    
    orchestrator_stats.finish();
    g_test_history.push_back(orchestrator_stats);
    
    // Print summary
    print_test_summary();
}

int main(int argc, char* argv[]) {
    // Disable core dumps to prevent core dump files
    struct rlimit core_limit;
    core_limit.rlim_cur = 0;
    core_limit.rlim_max = 0;
    setrlimit(RLIMIT_CORE, &core_limit);
    
    const long long RUN_DURATION_SECONDS = 120; // Reduced to 2 minutes for safety
    std::cout << "Starting ultimate Valgrind stress test..." << std::endl;
    std::cout << "Target duration: " << RUN_DURATION_SECONDS << " seconds." << std::endl;
    std::cout << "Core dumps disabled for safety." << std::endl;

    // Set up signal handler for graceful exit
    signal(SIGINT, [](int) { 
        std::cout << "\nReceived interrupt signal, shutting down gracefully..." << std::endl;
        g_should_exit.store(true); 
    });
    
    // Handle segmentation faults gracefully
    signal(SIGSEGV, [](int) {
        std::cout << "\nSegmentation fault caught, continuing with next test..." << std::endl;
        g_should_exit.store(true);
    });
    
    // Handle aborts gracefully
    signal(SIGABRT, [](int) {
        std::cout << "\nAbort signal caught, continuing with next test..." << std::endl;
        g_should_exit.store(true);
    });

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
        std::cout << "Running randomized test suites with enhanced timing and progress tracking." << std::endl;
        auto start_time = std::chrono::steady_clock::now();
        int cycle = 0;
        const int MAX_CYCLES = 50; // Hard limit to prevent infinite loops
        
        while (!g_should_exit.load() && cycle < MAX_CYCLES) {
            auto current_time = std::chrono::steady_clock::now();
            long long elapsed_seconds = std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time).count();
            
            if (elapsed_seconds >= RUN_DURATION_SECONDS) {
                std::cout << "\nTime limit reached, exiting..." << std::endl;
                break;
            }

            std::cout << "\n=========================================" << std::endl;
            std::cout << "Starting Enhanced Test Cycle " << ++cycle << "/" << MAX_CYCLES << std::endl;
            std::cout << "Elapsed: " << elapsed_seconds << "s / " << RUN_DURATION_SECONDS << "s" << std::endl;
            std::cout << "Tests completed so far: " << g_test_history.size() << std::endl;
            std::cout << "=========================================\n" << std::endl;
            
            // Add random delay between cycles
            std::uniform_int_distribution<int> delay_dist(100, 1000);
            std::this_thread::sleep_for(std::chrono::milliseconds(delay_dist(g_rng)));
            
            run_all_tests();
            
            // Print intermediate progress
            if (cycle % 5 == 0) {
                std::cout << "\n[INTERMEDIATE PROGRESS] After " << cycle << " cycles:" << std::endl;
                print_test_summary();
            }
        }
        
        if (cycle >= MAX_CYCLES) {
            std::cout << "\nMaximum cycle limit reached, exiting..." << std::endl;
        }
    }

    g_should_exit.store(true);
    std::cout << "\nUltimate stress test finished gracefully." << std::endl;
    
    // Final summary
    std::cout << "\n=== FINAL TEST EXECUTION REPORT ===" << std::endl;
    print_test_summary();
    
    return 0;
}
