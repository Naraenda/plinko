#include <cuda_runtime_api.h>
#include <driver_types.h>

#include <cub/block/block_reduce.cuh>
#include <cuda/std/bit>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>

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
#include <vector>


// char list is faster than compute
__device__ constexpr char chars[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

__device__ constexpr uint32_t k[64]{
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

__device__ constexpr uint32_t hash0[8]{0x6a09e667, 0xbb67ae85, 0x3c6ef372,
                                       0xa54ff53a, 0x510e527f, 0x9b05688c,
                                       0x1f83d9ab, 0x5be0cd19};

__device__ inline uint32_t ch(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (~x & z);
}

__device__ inline uint32_t ma(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (x & z) ^ (y & z);
}

template <uint32_t shift> __device__ inline uint32_t rotr(uint32_t a) {
    return cuda::std::rotr(a, shift);
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
#pragma unroll
        for (uint_fast8_t i = 0; i < num_bytes; ++i) {
            bytes[i] = value;
        }
    }

    __device__ __host__ inline byte_arr_t(const byte_arr_t &other) {
#pragma unroll
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
#pragma unroll
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
#pragma unroll
        for (uint32_t i = 0; i < num_ints; ++i) {
            if (ints[i] != other.ints[i])
                return false;
        }
        return true;
    }

    __device__ __host__ inline void flip_endianness() {
#pragma unroll
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

template <int First, int Last, typename Lambda>
__device__ inline void static_for(Lambda const &f) {
    if constexpr (First < Last) {
        f(cuda::std::integral_constant<int, First>{});
        static_for<First + 1, Last>(f);
    }
}

struct sha256_solver {
    uint32_t w0[16];
    uint32_t w[16];
    uint32_t z[8];

    __device__ void init() {
        for (uint_fast8_t i = 0; i < 8; ++i) {
            z[i] = hash0[i];
        }
    }

    template <uint32_t r> __device__ void expand(const message_t &m) {
        constexpr uint32_t j = r % 16;
        if (r < 16) {
            w0[j] = __byte_perm(m.ints[j], 0, 0x0123);
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

        uint32_t t1 = e1(e) + ch(e, f, g) + k[r] + w_r;
        uint32_t t2 = e0(a) + ma(a, b, c);

        h += t1;
        d += h;
        h += t2;
    }

    __device__ void get(hash_t &hash) {
        for (uint_fast8_t i = 0; i < 8; ++i) {
            uint32_t sum = hash0[i] + z[i];
            hash.ints[i] = __byte_perm(sum, 0, 0x0123);
        }
    }
};

template <uint32_t block_size, uint32_t hashes_per_thread>
__launch_bounds__(block_size) __global__
    void plinko(hash_t *global_hash, message_t *global_data,
                size_t global_offset = 0) {
    const uint32_t thread_id         = threadIdx.x;
    const uint32_t block_id          = blockIdx.x;
    const uint32_t threads_per_block = blockDim.x;

    hash_t local_hash{0xff};

    sha256_solver solver;

    message_t local_data;
    message_t candidate_data = global_data[0];

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
    //  7 |               x .
    //  8 |               . x
    // 20 |     x   x x   7 8          x
    //
    // we can only do 5 sha rounds though :(
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
        size_t thread_offset =
            i + (thread_id + block_id * threads_per_block) * hashes_per_thread;

        union {
            uint32_t ints[2];
            uint8_t  bytes[8];
        } payload;

        for (uint_fast8_t i = 0; i < 8; ++i) {
            payload.bytes[i] = chars[(thread_offset >> (6 * i)) % 64];
        }

        candidate_data.ints[7] = payload.ints[0];
        candidate_data.ints[8] = payload.ints[1];

        static_for<7, 9>([&](auto i) { solver.expand<i>(candidate_data); });
        static_for<7, 16>([&](auto i) { solver.round<i>(); });
        static_for<21, 32>([&](auto i) { solver.expand<i>(candidate_data); });
        static_for<16, 32>([&](auto i) { solver.round<i>(); });
        static_for<32, 48>([&](auto i) { solver.expand<i>(candidate_data); });
        static_for<32, 48>([&](auto i) { solver.round<i>(); });
        static_for<48, 64>([&](auto i) { solver.expand<i>(candidate_data); });
        static_for<48, 64>([&](auto i) { solver.round<i>(); });

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

    using hash_reduce_t = cub::BlockReduce<hash_t, block_size>;
    __shared__ typename hash_reduce_t::TempStorage reduce_storage;

    auto best_hash =
        hash_reduce_t{reduce_storage}.Reduce(local_hash, hash_t::min);

    if (thread_id == 0) {
        global_hash[block_id] = best_hash;
    }

    __syncthreads();

    if (global_hash[block_id] == local_hash) {
        global_data[block_id] = local_data;
    }
}

std::string return_current_time_and_date() {
    auto now       = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);

    std::stringstream ss;
    ss << std::put_time(std::gmtime(&in_time_t), "%Y%m%dT%H%M%S");
    return ss.str();
}

static_assert(sizeof(hash_t) == 32, "hash_t must be 32 bytes");
static_assert(sizeof(message_t) == 64, "message_t must be 64 bytes");

int32_t main() {
    std::string mask =
        // name/nonce               /RESERVED/  /yyyymmddTHHMMSS
        "Naraenda/trans-rights/uwu/+/________/++/_______________";

    message_t message('/');
    std::copy(mask.begin(), mask.end(), message.bytes);
    size_t message_size = 64 - 9;
    mask.resize(message_size, '/');

    std::cout << message.as_string(message_size) << std::endl;

    constexpr uint32_t grid_size  = 256;
    constexpr uint32_t block_size = 256;
    constexpr size_t   thread_size =
        size_t(64) * 64 * 64 * 64 * 64 * 64 / grid_size / block_size;

    std::vector<hash_t>    h_hashes(grid_size, hash_t{0xff});
    std::vector<message_t> h_messages(grid_size, message);

    message_t *d_messages;
    hash_t    *d_hashes;

    cudaMalloc(&d_hashes, sizeof(hash_t) * grid_size);
    cudaMalloc(&d_messages, sizeof(message_t) * grid_size);

    std::chrono::time_point<std::chrono::high_resolution_clock> t_now =
        std::chrono::high_resolution_clock::now();

    std::ofstream log_file;
    log_file.open("results.txt");

    hash_t    best_hash{0xff};
    message_t best_message{'?'};
    for (size_t offset = 0;; ++offset) {
        { // write time stamp
            auto timestamp = return_current_time_and_date();
            std::copy(timestamp.begin(), timestamp.end(), message.bytes + 40);
            message.pad(message_size);
            cudaMemcpy(d_messages, message.bytes, sizeof(message_t),
                       cudaMemcpyHostToDevice);
        }

        plinko<block_size, thread_size>
            <<<grid_size, block_size>>>(d_hashes, d_messages, offset);

        cudaMemcpy(h_messages.data(), d_messages, sizeof(message_t) * grid_size,
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(h_hashes.data(), d_hashes, sizeof(hash_t) * grid_size,
                   cudaMemcpyDeviceToHost);

        auto hash_candidate_ptr =
            std::min_element(h_hashes.begin(), h_hashes.end(), hash_t::less);

        if (*hash_candidate_ptr < best_hash) {
            best_hash = *hash_candidate_ptr;
            best_message =
                h_messages[std::distance(h_hashes.begin(), hash_candidate_ptr)];
            best_message.unpad(message_size);

            log_file << best_message.as_string(message_size) << std::endl;
        }

        auto t_prev = t_now;
        t_now       = std::chrono::high_resolution_clock::now();

        float dt =
            static_cast<std::chrono::duration<float>>(t_now - t_prev).count();
        size_t hashes_checked = block_size * grid_size * size_t{thread_size};

        std::cout << "\x1B[2J\x1B[H" << std::fixed << std::setw(11)
                  << std::setprecision(6) << hashes_checked / dt / 1e9
                  << " GH/sec\n"
                  << "input  : " << message.as_string(message_size) << "\n"
                  << "best   : " << best_message.as_string(message_size) << "\n"
                  << "offset : " << offset << "\n"
                  << "         ";
        for (uint32_t j = 0; j < sizeof(hash_t); ++j) {
            std::cout << std::hex << std::setw(2) << std::setfill('0')
                      << uint32_t{best_hash.bytes[j]};
        }
        std::cout << "\n" << std::endl;
    }

    cudaFree(d_hashes);
    cudaFree(d_messages);
}
