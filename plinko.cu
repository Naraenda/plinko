#include <cuda_runtime_api.h>
#include <driver_types.h>

#include <cub/block/block_reduce.cuh>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <ios>
#include <iostream>
#include <ostream>
#include <sstream>
#include <string>

// char list is faster than compute
__device__ constexpr char chars[] = "+0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ/";

__align__(32) __device__ constexpr uint32_t k[64]{
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
    0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
    0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
    0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
    0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
    0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
};

__align__(32) __device__
    constexpr uint32_t h0[8]{0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                             0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};

__device__ inline uint32_t ch(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (~x & z);
}

__device__ inline uint32_t ma(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (x & z) ^ (y & z);
}

template <uint32_t shift> __device__ inline uint32_t rotr(uint32_t a) {
    return a >> (shift % 32) | a << (32 - (shift % 32));
}

template <uint32_t a, uint32_t b, uint32_t c>
__device__ inline uint32_t e(uint32_t x) {
    return rotr<a>(x) ^ rotr<b>(x) ^ rotr<c>(x);
}

template <uint32_t a, uint32_t b, uint32_t c>
__device__ inline uint32_t s(uint32_t x) {
    return rotr<a>(x) ^ rotr<b>(x) ^ (x >> c);
}

__device__ inline uint32_t e0(uint32_t x) { return e<2, 13, 22>(x); }
__device__ inline uint32_t e1(uint32_t x) { return e<6, 11, 25>(x); }
__device__ inline uint32_t s0(uint32_t x) { return s<7, 18, 3>(x); }
__device__ inline uint32_t s1(uint32_t x) { return s<17, 19, 10>(x); }

template <uint32_t size> struct alignas(size) byte_arr_t {
    static constexpr uint_fast8_t num_bytes = size;
    static constexpr uint_fast8_t num_ints  = num_bytes / sizeof(uint32_t);

    union {
        uint8_t  bytes[num_bytes];
        uint32_t ints[num_ints];
    };

    __device__ __host__ inline byte_arr_t() {}

    __device__ __host__ inline byte_arr_t(const uint8_t value) {
        for (uint_fast8_t i = 0; i < num_bytes; ++i) {
            bytes[i] = value;
        }
    }

    __device__ __host__ inline byte_arr_t(const byte_arr_t &other) {
        for (uint_fast8_t i = 0; i < num_ints; ++i) {
            ints[i] = other.ints[i];
        }
    }

    __device__ __host__ static inline byte_arr_t &min(byte_arr_t &a,
                                                      byte_arr_t &b) {
        return a < b ? a : b;
    }

    __device__ __host__ static inline bool less(const byte_arr_t &a,
                                                const byte_arr_t &b) {
        for (uint_fast8_t i = 0; i < num_ints; ++i) {
            if (a.ints[i] == b.ints[i])
                continue;
            return a.ints[i] < b.ints[i];
        }
        return false;
    }

    __device__ __host__ inline bool operator<(const byte_arr_t &other) const {
        return byte_arr_t::less(*this, other);
    }

    __device__ __host__ inline bool operator==(const byte_arr_t &other) const {
        for (uint32_t i = 0; i < num_ints; ++i) {
            if (ints[i] != other.ints[i])
                return false;
        }
        return true;
    }

    __device__ inline void flip_endianness() {
        for (uint32_t i = 0; i < num_ints; ++i) {
            ints[i] = __byte_perm(ints[i], 0, 0x0123);
        }
    }

    std::string as_string(uint32_t chars = 48) {
        std::string s(chars, '?');
        std::copy(bytes, bytes + chars, s.begin());
        std::replace_if(
            s.begin(), s.end(), [](auto x) { return x < 0 || x > 127; }, '?');
        return s;
    }

    std::string as_hex_string() {
        std::stringstream ss;
        for (uint32_t i = 0; i < num_ints; ++i) {
            ss << std::hex << std::setw(8) << std::setfill('0') << ints[i]
               << " ";
        }
        return ss.str();
    }
};

using hash_t = byte_arr_t<32>;

struct message_t : byte_arr_t<64> {
    using byte_arr_t<64>::byte_arr_t;

    inline void pad(uint64_t byte_length) {
        uint64_t bit_length = byte_length * 8;

        for (uint32_t i = byte_length + 1; i < num_bytes; ++i) {
            bytes[i] = 0;
        }
        bytes[byte_length] = 0x80;

        for (uint32_t i = 0; i < 8; ++i) {
            bytes[63 - i] = bit_length >> (8 * i);
        }
    }

    inline void unpad(uint64_t byte_length) {
        bytes[byte_length] &= ~0x80;
        ints[15] = 0;
        ints[14] = 0;
    }
};

template<class T, T v>
struct integral_constant
{
    static constexpr T value = v;
    using value_type = T;
    using type = integral_constant; // using injected-class-name
    __host__ __device__ constexpr operator value_type() const noexcept { return value; }
    __host__ __device__ constexpr value_type operator()() const noexcept { return value; } // since c++14
};

template <uint32_t First, uint32_t Last, typename Lambda>
__device__ inline void static_for(Lambda const& f)
{
    if constexpr (First < Last)
    {
        f(integral_constant<uint32_t, First>{});
        static_for<First + 1, Last>(f);
    }
}

struct sha256_solver {
    uint32_t w0[16];
    uint32_t w[16];
    uint32_t z[8];

    __device__ void init() {
        for (uint_fast8_t i = 0; i < 8; ++i) {
            z[i] = h0[i];
        }
    }

    template <uint32_t r> __device__ void expand(const message_t &message) {
        constexpr uint32_t j = r % 16;
        if (r < 16) {
            w0[j] = message.ints[j];
        } else {
            const auto w_ = [&](auto i) {
                return i < 32 ? w0[i % 16] : w[i % 16];
            };

            w[j] = s1(w_(r + 14)) + w_(r + 9) + s0(w_(r + 1)) + w_(r);
        }
    }

    template <uint32_t r> __device__ void round() {
        uint32_t &a = z[(64 + 0 - r) % 8];
        uint32_t &b = z[(64 + 1 - r) % 8];
        uint32_t &c = z[(64 + 2 - r) % 8];
        uint32_t &d = z[(64 + 3 - r) % 8];
        uint32_t &e = z[(64 + 4 - r) % 8];
        uint32_t &f = z[(64 + 5 - r) % 8];
        uint32_t &g = z[(64 + 6 - r) % 8];
        uint32_t &h = z[(64 + 7 - r) % 8];

        const uint32_t w_r = r < 16 ? w0[r % 16] : w[r % 16];

        const uint32_t t1 = e1(e) + ch(e, f, g) + k[r] + w_r;
        const uint32_t t2 = e0(a) + ma(a, b, c);

        h += t1;
        d += h;
        h += t2;
    }

    __device__ void get(hash_t &hash) {
        for (uint_fast8_t i = 0; i < 8; ++i) {
            hash.ints[i] = h0[i] + z[i];
        }
    }
};

template <uint32_t block_size, uint32_t hashes_per_thread,
          uint32_t slow_retries>
__launch_bounds__(block_size) __global__
    void plinko(hash_t *global_hash, message_t *global_data) {
    const uint32_t thread_id         = threadIdx.x;
    const uint32_t block_id          = blockIdx.x;
    const uint32_t threads_per_block = blockDim.x;
    const uint32_t global_thread_id  = thread_id + block_id * threads_per_block;

    hash_t    local_hash{0xff};
    message_t local_data;
    message_t candidate_data = global_data[0];

    candidate_data.flip_endianness();

    sha256_solver solver;
    for (uint_fast8_t k = 0; k < slow_retries; ++k) {
        candidate_data.bytes[0x19] = chars[k];

        // we can pre compute some of the expanded 'w'
        // and a few hash rounds if we limit the values
        // we mutate for hash generation
        //
        //    |             w[_] (calculated via expand)
        //  R | 0 1 2 3 4 5 6 7 8 9 a b c d e f
        // 16 | x x           . . x         x
        // 17 |   x x         . .   x         x
        // 18 | x   x x       . .     x
        // 19 |   x   x x     . .       x
        // 20 |     x   x x   . .         x
        // 21 |       x   x x . .           x
        // =========== MUTATE 7 8 ================
        //  7 |               x .   required for sha
        //  8 |               . x   round 7 and 8
        //
        // we can only do 6 sha rounds though :(
        //
        // 2 ints at index 7, 8
        // search space is 2^32 different hashes
        // at 20GH/sec is like a few seconds

        solver.init();
        static_for<0, 21>([&](auto i) {
            if constexpr (i < 7 || 9 <= i)
                solver.expand<i>(candidate_data);
        });
        static_for<0, 7>([&](auto i) { solver.round<i>(); });

        // backup solver state
        uint32_t w1[32];
        uint32_t z1[8];

        for (uint_fast8_t i = 0; i < 32; ++i) {
            w1[i] = solver.w[i];
        }
        for (uint_fast8_t i = 0; i < 8; ++i) {
            z1[i] = solver.z[i];
        }

        // brute force loop
        for (uint32_t i = 0; i < hashes_per_thread; ++i) {
            const uint64_t global_hash_id =
                i + global_thread_id * static_cast<uint64_t>(hashes_per_thread);

            union {
                uint32_t ints[2];
                uint8_t  bytes[8];
            } payload;

            for (uint_fast8_t i = 0; i < 8; ++i) {
                payload.bytes[i] = chars[(global_hash_id >> (6 * i)) % 64];
            }

            candidate_data.ints[7] = payload.ints[0];
            candidate_data.ints[8] = payload.ints[1];
            static_for<7, 9>([&](auto i) { solver.expand<i>(candidate_data); });
            static_for<7, 21>([&](auto i) { solver.round<i>(); });
            static_for<21, 64>([&](auto i) {
                solver.expand<i>(candidate_data);
                solver.round<i>();
            });

            hash_t candidate_hash;
            solver.get(candidate_hash);

            // candidate is better
            if (candidate_hash < local_hash) {
                local_hash = candidate_hash;
                local_data = candidate_data;
            }

            // rollback solver
            for (uint_fast8_t i = 0; i < 32; ++i) {
                solver.w[i] = w1[i];
            }
            for (uint_fast8_t i = 0; i < 8; ++i) {
                solver.z[i] = z1[i];
            }
        }
    }

    using hash_reduce_t = cub::BlockReduce<hash_t, block_size>;
    __shared__ typename hash_reduce_t::TempStorage reduce_storage;

    auto best_local_hash =
        hash_reduce_t{reduce_storage}.Reduce(local_hash, hash_t::min);

    __shared__ hash_t best_shared_hash;
    if (thread_id == 0) {
        best_shared_hash = best_local_hash;
    }

    __syncthreads();

    if (best_shared_hash == local_hash) {
        global_hash[block_id] = local_hash;
        local_data.flip_endianness();
        global_data[block_id] = local_data;
    }
}

std::string get_timestamp() {
    auto now = std::chrono::system_clock::now();
    // Convert to time_t which is a time point
    std::time_t now_time_t = std::chrono::system_clock::to_time_t(now);
    // Convert to tm structure
    std::tm           now_tm = *std::gmtime(&now_time_t);
    std::stringstream ss;
    ss << std::put_time(&now_tm, "%y-%m-%d/%H+%M+%S/");
    return ss.str();
}

static_assert(sizeof(hash_t) == 32, "hash_t must be 32 bytes");
static_assert(sizeof(message_t) == 64, "message_t must be 64 bytes");

struct plinko_box {
    static constexpr uint64_t search_range = 16 * (1ULL << (4 * 8));
    static constexpr uint32_t grid_size    = 256;
    static constexpr uint32_t block_size   = 512;
    static constexpr uint32_t slow_retries = 1; // use this if you run out of search range

    static_assert(search_range <= (1ULL << 6 * 8),
                  "search range must not be greater than all possible "
                  "character combinations");
    static constexpr size_t thread_size = search_range / grid_size / block_size;

    static constexpr size_t message_size = 64 - 9;

    message_t  seed_message;
    message_t  best_message;
    hash_t     best_hash;
    hash_t     h_hashes[grid_size];
    message_t  h_messages[grid_size];
    hash_t    *d_hashes;
    message_t *d_messages;

    cudaStream_t  stream;
    std::ofstream log_file;

    std::chrono::time_point<std::chrono::high_resolution_clock> t_now =
        std::chrono::high_resolution_clock::now();

    plinko_box &init(std::string mask) {
        log_file.open("results.txt", std::ios_base::app);

        best_hash    = hash_t(0xff);
        seed_message = message_t('0');

        std::copy(mask.begin(), mask.end(), seed_message.bytes);

        return *this;
    }

    void prepare_next() {
        auto timestamp = get_timestamp();
        std::copy(timestamp.begin(), timestamp.end(), seed_message.bytes + 37);
        seed_message.pad(message_size);
    }

    void check_previous() {
        auto hash_candidate_ptr =
            std::min_element(h_hashes, h_hashes + grid_size, hash_t::less);

        if (*hash_candidate_ptr < best_hash) {
            best_hash = *hash_candidate_ptr;
            best_message =
                h_messages[std::distance(h_hashes, hash_candidate_ptr)];
            best_message.unpad(message_size);

            log_file << best_message.as_string(message_size) << "\n"
                     << best_hash.as_hex_string() << std::endl;
        }

        auto t_prev = t_now;
        t_now       = std::chrono::high_resolution_clock::now();

        float dt =
            static_cast<std::chrono::duration<float>>(t_now - t_prev).count();
        uint64_t hashes_checked =
            slow_retries * block_size * grid_size * uint64_t{thread_size};

        std::cout << "\x1B[2J\x1B[H" << std::fixed << std::setw(11)
                  << std::setprecision(6) << hashes_checked / dt / 1e9
                  << " GH/sec\n"
                  << "input  : " << seed_message.as_string(message_size) << "\n"
                  << "best   : " << best_message.as_string(message_size) << "\n"
                  << "         " << best_hash.as_hex_string() << std::endl;
    }

    void run() {
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

        cudaMallocAsync(&d_hashes, sizeof(hash_t) * grid_size, stream);
        cudaMallocAsync(&d_messages, sizeof(message_t) * grid_size, stream);

        cudaEvent_t results_ready;
        cudaEventCreate(&results_ready);

        prepare_next();
        for (uint64_t i = 0;; ++i) {
            // copy pure input to device
            cudaMemcpyAsync(d_messages, &seed_message, sizeof(message_t),
                            cudaMemcpyHostToDevice, stream);
            // horse plinko
            plinko<block_size, thread_size, slow_retries>
                <<<grid_size, block_size, 0, stream>>>(d_hashes, d_messages);

            // we are computing results on device, do stuff on host in parallel
            if (i > 0) {
                // wait for device results to be copied
                cudaEventSynchronize(results_ready);
                prepare_next();
                check_previous();
            }

            // copy mutated input to host
            cudaMemcpyAsync(h_messages, d_messages,
                            sizeof(message_t) * grid_size,
                            cudaMemcpyDeviceToHost, stream);
            // copy hashes to host
            cudaMemcpyAsync(h_hashes, d_hashes, sizeof(hash_t) * grid_size,
                            cudaMemcpyDeviceToHost, stream);
            // we need to know when results have been copied over
            cudaEventRecord(results_ready, stream);
            cudaDeviceSynchronize();
        }
    }
};

int32_t main() {
    std::string mask =
        "Naraenda/trans-rights/uwu/+/________/_________________/";
    //     write your thingies here /RESERVED/yy-mm-dd/hh+dd+ss/
    //   0123456789abcdef0123456789abcdef0123456789abcdef0123456
    plinko_box{}.init(mask).run();
}
