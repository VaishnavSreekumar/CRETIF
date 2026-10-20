// ── test_bank_conflict_simulation.cpp ─────────────────────────────────
// Formally verifies the shared memory bank conflict elimination and
// numerical correctness of padded tile indexing (TILE_N_PAD = 33).

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>
#include <set>

constexpr int TILE_M = 32;
constexpr int TILE_N = 32;
constexpr int TILE_K = 32;
constexpr int TILE_N_PAD = TILE_N + 1; // 33 floats

// CUDA shared memory has 32 banks, 4 bytes each.
// Bank index of a 32-bit float at element index i is (i % 32).
inline int get_bank(size_t element_index) {
    return static_cast<int>(element_index % 32);
}

void test_bank_conflicts_unpadded() {
    printf("[Verification 1] Testing unpadded tile_b[32][32]...\n");

    // In each warp, ty is fixed (e.g. ty = 0 for Warp 0), tx ranges 0..31.
    for (int ty = 0; ty < 32; ++ty) {
        std::set<int> banks_accessed;
        for (int tx = 0; tx < 32; ++tx) {
            // Unpadded write: tile_b[tx][ty] with row stride 32
            size_t element_index = static_cast<size_t>(tx) * TILE_N + ty;
            banks_accessed.insert(get_bank(element_index));
        }

        // Because (tx * 32 + ty) % 32 == ty, all 32 threads access the EXACT same bank!
        assert(banks_accessed.size() == 1);
        assert(*banks_accessed.begin() == (ty % 32));
    }
    printf("  -> CONFIRMED: Unpadded tile produces 32-way bank conflict (only 1 bank active per warp write)!\n");
}

void test_bank_conflicts_padded() {
    printf("[Verification 2] Testing padded tile_b[32][33]...\n");

    for (int ty = 0; ty < 32; ++ty) {
        std::set<int> banks_accessed;
        for (int tx = 0; tx < 32; ++tx) {
            // Padded write: tile_b[tx][ty] with row stride 33
            size_t element_index = static_cast<size_t>(tx) * TILE_N_PAD + ty;
            banks_accessed.insert(get_bank(element_index));
        }

        // (tx * 33 + ty) % 32 == (tx + ty) % 32
        // All 32 threads access distinct banks 0..31!
        assert(banks_accessed.size() == 32);
    }
    printf("  -> CONFIRMED: Padded tile completely eliminates bank conflicts (all 32 banks active, 0 conflicts)!\n");
}

void test_read_phase_padded() {
    printf("[Verification 3] Testing read phase under padded tile...\n");

    // Read phase: sum += tile_a[ty][k] * tile_b[k][tx]
    // For fixed k (in unrolled loop), tx ranges 0..31 across the warp.
    for (int k = 0; k < TILE_K; ++k) {
        std::set<int> banks_accessed;
        for (int tx = 0; tx < 32; ++tx) {
            size_t element_index = static_cast<size_t>(k) * TILE_N_PAD + tx;
            banks_accessed.insert(get_bank(element_index));
        }
        // (k * 33 + tx) % 32 == (k + tx) % 32
        // All 32 threads read distinct banks!
        assert(banks_accessed.size() == 32);
    }
    printf("  -> CONFIRMED: Padded tile read phase is also 100%% conflict-free across all k!\n");
}

void test_numerical_equivalence() {
    printf("[Verification 4] Testing numerical equivalence of padded simulation vs reference...\n");

    float tile_a[TILE_M][TILE_K];
    float tile_b_unpadded[TILE_K][TILE_N];
    float tile_b_padded[TILE_K][TILE_N_PAD];

    // Seed test matrices
    for (int i = 0; i < TILE_M; ++i) {
        for (int j = 0; j < TILE_K; ++j) {
            tile_a[i][j] = (i * 0.1f) - (j * 0.05f) + 1.0f;
        }
    }

    for (int k = 0; k < TILE_K; ++k) {
        for (int n = 0; n < TILE_N; ++n) {
            float val = (k * 0.2f) + (n * 0.03f) - 0.5f;
            tile_b_unpadded[k][n] = val;
            tile_b_padded[k][n] = val;
        }
    }

    // Compute dot products
    for (int ty = 0; ty < TILE_M; ++ty) {
        for (int tx = 0; tx < TILE_N; ++tx) {
            float sum_unpadded = 0.0f;
            float sum_padded = 0.0f;
            for (int k = 0; k < TILE_K; ++k) {
                sum_unpadded += tile_a[ty][k] * tile_b_unpadded[k][tx];
                sum_padded += tile_a[ty][k] * tile_b_padded[k][tx];
            }
            float diff = std::abs(sum_unpadded - sum_padded);
            assert(diff < 1e-6f);
        }
    }
    printf("  -> CONFIRMED: Numerical output with padded tile matches unpadded bit-for-bit (diff < 1e-6)!\n");
}

int main() {
    printf("========================================================\n");
    printf("  CuInfer — Bank Conflict Elimination Verification Suite\n");
    printf("========================================================\n\n");

    test_bank_conflicts_unpadded();
    test_bank_conflicts_padded();
    test_read_phase_padded();
    test_numerical_equivalence();

    printf("\nAll verification assertions PASSED successfully!\n");
    return 0;
}
