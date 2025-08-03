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
static const bool VERBOSE = false;
// ====================================================================
// PART 1: MEMORY ABUSER - Triggers for Valgrind's Memcheck
// ====================================================================
namespace MemoryAbuser {

void trigger_use_after_free() {
    if (VERBOSE) std::cout << "[Mem] Triggering use-after-free..." << std::endl;
    int* ptr = new int[50];
    ptr[25] = 123;
    delete[] ptr;
    // ERROR: Using freed memory. Valgrind will detect this read.
    // We suppress the output as it might crash, but Valgrind sees the read.
    volatile int val = ptr[25]; 
    (void)val; // Prevent compiler from optimizing away the read.
}

void trigger_heap_overflow() {
    if (VERBOSE) std::cout << "[Mem] Triggering heap buffer overflow..." << std::endl;
    char* buffer = new char[32];
    // ERROR: Writing 1 byte past the allocated chunk.
    buffer[32] = 'X'; 
    delete[] buffer;
}

void trigger_stack_overflow() {
    if (VERBOSE) std::cout << "[Mem] Triggering stack buffer overflow..." << std::endl;
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
    if (VERBOSE) std::cout << "[Mem] Triggering uninitialized value read..." << std::endl;
    int x; // Uninitialized
    // ERROR: x is uninitialized and used in a conditional jump.
    if (x > 100) {
        x = 1;
    } else {
        x = 0;
    }
    if (VERBOSE) std::cout << "[Mem] Uninitialized value was used in a branch (result: " << x << ")." << std::endl;
}

void trigger_memory_leak() {
    if (VERBOSE) std::cout << "[Mem] Triggering a definite memory leak..." << std::endl;
    // ERROR: Leaking this memory.
    new std::string("This string is a definite memory leak.");
}

void trigger_mismatched_free() {
    if (VERBOSE) std::cout << "[Mem] Triggering mismatched new[]/delete..." << std::endl;
    // ERROR: Allocating an array but using scalar delete.
    int* array = new int[10];
    delete array;
}

void trigger_double_free() {
    if (VERBOSE) std::cout << "[Mem] Triggering double free..." << std::endl;
    int* p = new int(42);
    delete p;
    // ERROR: Deleting the same memory twice.
    // This is very likely to crash, so we'll comment it out for now
    // and let Valgrind detect the other errors instead
    // delete p;
    if (VERBOSE) std::cout << "[Mem] Double free skipped to prevent crash." << std::endl;
}

void trigger_overlapping_memcpy() {
    if (VERBOSE) std::cout << "[Mem] Triggering overlapping src/dst in memcpy..." << std::endl;
    char data[20];
    strcpy(data, "0123456789");
    // ERROR: src and dst overlap in a way that memcpy doesn't support.
    // Use memmove instead of memcpy for actual safety, but memcpy for Valgrind detection
    // Reduce overlap to be less likely to crash
    memcpy(data + 1, data, 8); // Smaller overlap
}

void run() {
    if (VERBOSE) std::cout << "\n--- Running Memory Abuser ---" << std::endl;
    
    // Wrap dangerous operations in try-catch to prevent crashes
    try {
        trigger_use_after_free();
    } catch (...) {
        if (VERBOSE) std::cout << "[Mem] Use-after-free caught exception." << std::endl;
    }
    
    try {
        trigger_heap_overflow();
    } catch (...) {
        if (VERBOSE) std::cout << "[Mem] Heap overflow caught exception." << std::endl;
    }
    
    try {
        trigger_stack_overflow();
    } catch (...) {
        if (VERBOSE) std::cout << "[Mem] Stack overflow caught exception." << std::endl;
    }
    
    try {
        trigger_uninitialized_read();
    } catch (...) {
        if (VERBOSE) std::cout << "[Mem] Uninitialized read caught exception." << std::endl;
    }
    
    try {
        trigger_memory_leak();
    } catch (...) {
        if (VERBOSE) std::cout << "[Mem] Memory leak caught exception." << std::endl;
    }
    
    try {
        trigger_mismatched_free();
    } catch (...) {
        if (VERBOSE) std::cout << "[Mem] Mismatched free caught exception." << std::endl;
    }
    
    try {
        trigger_double_free();
    } catch (...) {
        if (VERBOSE) std::cout << "[Mem] Double free caught exception." << std::endl;
    }
    
    try {
        trigger_overlapping_memcpy();
    } catch (...) {
        if (VERBOSE) std::cout << "[Mem] Overlapping memcpy caught exception." << std::endl;
    }
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
    if (VERBOSE) std::cout << "[Thread] Deadlock thread A starting..." << std::endl;
    
    auto start = std::chrono::steady_clock::now();
    if (g_mutex_a.try_lock()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        if (VERBOSE) std::cout << "[Thread] A trying to get lock B..." << std::endl;
        
        // Check for timeout
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - start).count();
        if (elapsed > 5 || g_should_exit.load()) {
            if (VERBOSE) std::cout << "[Thread] A timed out." << std::endl;
            g_mutex_a.unlock();
            return;
        }
        
        if (g_mutex_b.try_lock()) {
            if (VERBOSE) std::cout << "[Thread] A acquired both locks." << std::endl;
            g_mutex_b.unlock();
        } else {
            if (VERBOSE) std::cout << "[Thread] A couldn't get lock B." << std::endl;
        }
        g_mutex_a.unlock();
    } else {
        if (VERBOSE) std::cout << "[Thread] A couldn't get lock A." << std::endl;
    }
}

void deadlock_thread_b() {
    if (g_should_exit.load()) return;
    if (VERBOSE) std::cout << "[Thread] Deadlock thread B starting..." << std::endl;
    
    auto start = std::chrono::steady_clock::now();
    if (g_mutex_b.try_lock()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        if (VERBOSE) std::cout << "[Thread] B trying to get lock A..." << std::endl;
        
        // Check for timeout
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - start).count();
        if (elapsed > 5 || g_should_exit.load()) {
            if (VERBOSE) std::cout << "[Thread] B timed out." << std::endl;
            g_mutex_b.unlock();
            return;
        }
        
        if (g_mutex_a.try_lock()) {
            if (VERBOSE) std::cout << "[Thread] B acquired both locks." << std::endl;
            g_mutex_a.unlock();
        } else {
            if (VERBOSE) std::cout << "[Thread] B couldn't get lock A." << std::endl;
        }
        g_mutex_b.unlock();
    } else {
        if (VERBOSE) std::cout << "[Thread] B couldn't get lock B." << std::endl;
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
    if (VERBOSE) std::cout << "[Thread] Spawning threads for data race..." << std::endl;
    std::vector<std::thread> threads;
    for (int i = 0; i < 4; ++i) threads.emplace_back(racy_thread_func);
    for (auto& t : threads) t.join();
    if (VERBOSE) std::cout << "[Thread] Final racy value (expected " << 200000 * 4 << "): " << g_shared_racy_data << std::endl;
}

void run_deadlock() {
    if (VERBOSE) std::cout << "[Thread] Spawning threads for deadlock prevention test..." << std::endl;
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
            if (VERBOSE) std::cout << "[Thread] Deadlock test timed out, continuing..." << std::endl;
            break;
        }
    }
}

void run_producer_consumer() {
    if (VERBOSE) std::cout << "[Thread] Running producer-consumer chaos..." << std::endl;
    ProducerConsumerChaos::g_done = false;
    std::vector<std::thread> threads;
    threads.emplace_back(ProducerConsumerChaos::producer);
    threads.emplace_back(ProducerConsumerChaos::consumer);
    threads.emplace_back(ProducerConsumerChaos::consumer);
    for(auto& t : threads) t.join();
}

void run() {
    if (VERBOSE) std::cout << "\n--- Running Threading Nightmare ---" << std::endl;
    
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
    if (VERBOSE) std::cout << "[Syscall] Fork and random crash signal test..." << std::endl;
    
    pid_t pid = fork();
    if (pid == -1) { 
        perror("fork"); 
        return; 
    }

    if (pid == 0) { // Child process
        if (VERBOSE) std::cout << "[Child] Child process started, doing random work..." << std::endl;
        
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
        
        if (VERBOSE) std::cout << "[Child] Child completed work, sum = " << sum << std::endl;
        _exit(0);
    } else { // Parent process
        // Give child a moment to start its work
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        
        // Send a random crash signal to the child
        std::vector<int> crash_signals = {SIGTERM, SIGKILL, SIGINT, SIGUSR1, SIGUSR2};
        std::uniform_int_distribution<size_t> signal_dist(0, crash_signals.size() - 1);
        int chosen_signal = crash_signals[signal_dist(g_rng)];
        
        if (VERBOSE) std::cout << "[Parent] Sending signal " << chosen_signal << " to child process " << pid << std::endl;
        
        if (kill(pid, chosen_signal) == -1) {
            perror("kill");
        }
        
        // Wait for child and get its exit status
        int status;
        pid_t result = waitpid(pid, &status, 0);
        if (result != -1) {
            if (WIFEXITED(status)) {
                if (VERBOSE) std::cout << "[Parent] Child exited normally with code " << WEXITSTATUS(status) << std::endl;
            } else if (WIFSIGNALED(status)) {
                if (VERBOSE) std::cout << "[Parent] Child terminated by signal " << WTERMSIG(status) << std::endl;
            }
        } else {
            perror("waitpid");
        }
    }
}

void run() {
    if (VERBOSE) std::cout << "\n--- Running IPC and Syscall Chaos ---" << std::endl;
    
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
                if (VERBOSE) std::cout << "[IPC] Parent received: '" << buffer << "'" << std::endl;
                close(pipe_fd[0]);
                wait(NULL);
            }
        },
        run_crash_signal_test,
        []() {
            if (VERBOSE) std::cout << "[Syscall] Leaking a file descriptor..." << std::endl;
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

// --- 4a: Graph algorithm (Dijkstra's) ---
void run_dijkstra() {
    if (VERBOSE) std::cout << "[Algo] Running Dijkstra's Algorithm..." << std::endl;
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
    if (VERBOSE) std::cout << "[Algo] Running N-Queens Solver (N="<< n << ")..." << std::endl;
    std::vector<int> board(n);
    NQueens::solve(0, board); // Stresses call stack
}

// --- 4c: Numerical Computation (Matrix Multiplication) ---
void run_matrix_multiplication() {
    if (VERBOSE) std::cout << "[Algo] Running Matrix Multiplication..." << std::endl;
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
    if (VERBOSE) std::cout << "\n--- Running Advanced Algorithm Stress Test ---" << std::endl;
    
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
    if (VERBOSE) std::cout << "[Algo] Running Red-Black Tree operations..." << std::endl;
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
    if (VERBOSE) std::cout << "[Algo] Running Fast Fourier Transform..." << std::endl;
    std::vector<std::complex<double>> signal(1024);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    
    for (auto& sample : signal) {
        sample = std::complex<double>(dist(g_rng), 0);
    }
    
    FFT::fft(signal);
}

// --- 4f: Traveling Salesman Problem (Dynamic Programming) ---
void run_tsp() {
    if (VERBOSE) std::cout << "[Algo] Running TSP with Dynamic Programming..." << std::endl;
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
    if (VERBOSE) std::cout << "[Algo] Running Suffix Array construction..." << std::endl;
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

// --- 5a: Heap profiler target (Massif) ---
void run_massif_sawtooth() {
    if (VERBOSE) std::cout << "[Profile] Running Massif 'sawtooth' heap stress..." << std::endl;
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
    if (VERBOSE) std::cout << "[Profile] Running Cachegrind stress (naive vs. tiled)..." << std::endl;
    // Initializing matrix data once
    for(int i=0; i<CacheStress::SIZE; ++i) for(int j=0; j<CacheStress::SIZE; ++j) CacheStress::matrix[i][j] = i*j;

    // Run the inefficient version
    CacheStress::transpose_naive();
    
    // Run the efficient version. A profiler like Cachegrind will show
    // a dramatic reduction in cache misses (D1 misses, L3 misses).
    CacheStress::transpose_tiled(32);
}

void run() {
    if (VERBOSE) std::cout << "\n--- Running Advanced Profiling Targets ---" << std::endl;
    run_massif_sawtooth();
    run_cache_stress();
}

} // namespace ProfilingTargets


// ====================================================================
// --- MAIN ORCHESTRATOR ---
// ====================================================================
void run_all_tests() {
    // Randomize the order of test execution
    std::vector<std::function<void()>> test_functions = {
        MemoryAbuser::run,
        ThreadingNightmare::run,
        SyscallChaos::run,
        AlgorithmStress::run,
        ProfilingTargets::run
    };
    
    std::shuffle(test_functions.begin(), test_functions.end(), g_rng);
    
    // Run a random subset of tests
    std::uniform_int_distribution<int> count_dist(2, test_functions.size());
    int num_to_run = count_dist(g_rng);
    
    for (int i = 0; i < num_to_run; ++i) {
        if (g_should_exit.load()) break;
        test_functions[i]();
    }
}

int main(int argc, char* argv[]) {
    // Disable core dumps to prevent core dump files
    struct rlimit core_limit;
    core_limit.rlim_cur = 0;
    core_limit.rlim_max = 0;
    setrlimit(RLIMIT_CORE, &core_limit);
    
    const long long RUN_DURATION_SECONDS = 120; // Reduced to 2 minutes for safety
    if (VERBOSE) std::cout << "Starting ultimate Valgrind stress test..." << std::endl;
    if (VERBOSE) std::cout << "Target duration: " << RUN_DURATION_SECONDS << " seconds." << std::endl;
    if (VERBOSE) std::cout << "Core dumps disabled for safety." << std::endl;

    // Set up signal handler for graceful exit
    signal(SIGINT, [](int) { 
        if (VERBOSE) std::cout << "\nReceived interrupt signal, shutting down gracefully..." << std::endl;
        g_should_exit.store(true); 
    });
    
    // Handle segmentation faults gracefully
    signal(SIGSEGV, [](int) {
        if (VERBOSE) std::cout << "\nSegmentation fault caught, continuing with next test..." << std::endl;
        g_should_exit.store(true);
    });
    
    // Handle aborts gracefully
    signal(SIGABRT, [](int) {
        if (VERBOSE) std::cout << "\nAbort signal caught, continuing with next test..." << std::endl;
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
            if (VERBOSE) std::cout << "Running only the '" << arg << "' test suite." << std::endl;
            test_suites[arg]();
        } else {
            std::cerr << "Unknown test suite: " << arg << std::endl;
            std::cerr << "Available suites: memory, threads, syscall, algo, profile, all" << std::endl;
            return 1;
        }
    } else {
        if (VERBOSE) std::cout << "Running randomized test suites in a time-limited loop." << std::endl;
        auto start_time = std::chrono::steady_clock::now();
        int cycle = 0;
        const int MAX_CYCLES = 50; // Hard limit to prevent infinite loops
        
        while (!g_should_exit.load() && cycle < MAX_CYCLES) {
            auto current_time = std::chrono::steady_clock::now();
            long long elapsed_seconds = std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time).count();
            
            if (elapsed_seconds >= RUN_DURATION_SECONDS) {
                if (VERBOSE) std::cout << "\nTime limit reached, exiting..." << std::endl;
                break;
            }

            if (VERBOSE) std::cout << "\n=========================================" << std::endl;
            if (VERBOSE) std::cout << "Starting Test Cycle " << ++cycle << "/" << MAX_CYCLES << " (Elapsed: " << elapsed_seconds << "s)" << std::endl;
            if (VERBOSE) std::cout << "=========================================\n" << std::endl;
            
            // Add random delay between cycles
            std::uniform_int_distribution<int> delay_dist(100, 1000);
            std::this_thread::sleep_for(std::chrono::milliseconds(delay_dist(g_rng)));
            
            run_all_tests();
        }
        
        if (cycle >= MAX_CYCLES) {
            if (VERBOSE) std::cout << "\nMaximum cycle limit reached, exiting..." << std::endl;
        }
    }

    g_should_exit.store(true);
    if (VERBOSE) std::cout << "\nUltimate stress test finished gracefully." << std::endl;
    return 0;
}