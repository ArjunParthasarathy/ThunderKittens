// #define TORCH_COMPILE
#include "kittens.cuh"
#include "prototype.cuh"

#define NUM_WORKERS 1
#define NUM_WARPS 1
#define NUM_THREADS (NUM_WARPS * kittens::WARP_THREADS)

// shared patterns
#define SQRT_N 64
#define rt_cmplx_bf_base crt_bf<SQRT_N, SQRT_N>
#define rt_cmplx_bf_base_col crt_bf<SQRT_N, SQRT_N, ducks::rt_layout::col>
#define rt_cmplx_fl_base crt_fl<SQRT_N, SQRT_N>
#define st_cmplx_bf_base cst_bf<SQRT_N, SQRT_N>

using namespace kittens;
template<int n1> struct fftconv_layout {
    using seq_layout = gl<bf16, -1, -1, n1, n1>;
    using filter_layout = cgl<gl<bf16, 1, -1, 64, 64>>;
    using fft_layout = cgl<gl<bf16, 1, 1, 64, 64>>;

    // using complex_input_layout = kittens::cgl<input_layout>;
    // using complex_filter_layout = kittens::cgl<filter_layout>;
    // using complex_fft_layout = kittens::cgl<fft_layout>;
    
    struct globals { 
        seq_layout u_g;
        seq_layout o_real_g;
        filter_layout kf_g;
        fft_layout f_g, finv_g, tw_g, twinv_g;
    };
};
template<int _n, int _n1>
struct fftconv_template {
    static constexpr int n=_n, n1=_n1;
    using layout = fftconv_layout<n1>;
};

template<typename T>
__global__ void fftconv_tk(typename T::layout::globals g, int B_TILE, int H_TILE) {
    int warpid = kittens::warpid();
    constexpr int N = T::n;
    constexpr int N1 = T::n1;
    // Number of subtiles (sub-batches in each 64x64 tiles)
    int ST = (N + N1 - 1) / (N1*N1);
    // Number of rows/cols of subtiles (sqrt of ST)
    // int rows = 0;
    // if (ST == 16) {
    //     rows = 4;
    // }
    // else if (ST == 32) {
    //     rows = 2;
    // } else {
    //     rows = 1;
    // }

    int b_start = blockIdx.x * B_TILE;

    // Registers; everyone loads
    rt_cmplx_bf_base a_reg;       
    rt_cmplx_fl_base mma_reg;     
    rt_cmplx_bf_base accum;       
    rt_cmplx_bf_base_col b_reg;

    zero(a_reg);
    zero(mma_reg);
    zero(accum);
    zero(b_reg);

    for (int i = 0; i < H_TILE; i++) {
        for (int j = b_start; j < b_start+B_TILE; j++) {   
            // In code we load in by quadrants of the batches
            // For each outer batch, we load in the inner batches, and then do the computation of 
            int diff = j - b_start;
            auto b_st = subtile_inplace<N1>(b_reg.real, diff % ST);
            load(b_st, g.u_g, {j, i, 0, 0});
            // X = F^T X
            load(a_reg, g.f_g, {0, 0, 0, 0});
            transpose_inplace(a_reg);
            kittens::zero(mma_reg);
            kittens::mma_AB(mma_reg, a_reg, b_reg, mma_reg);
            kittens::copy(accum, mma_reg);

            // X = X * tw
            load(a_reg, g.tw_g, {0, 0, 0, 0});// needs to be imag too.
            kittens::mul(accum, accum, a_reg);

            // // X = XF
            load(b_reg, g.f_g, {0, 0, 0, 0}); // needs to be imag too.
            kittens::zero(mma_reg);
            kittens::mma_AB(mma_reg, accum, b_reg, mma_reg);
            kittens::copy(accum, mma_reg);

            // X = X * K_f^T
            load(a_reg, g.kf_g, {0, i, 0, 0});
            kittens::mul(accum, accum, a_reg);

            // X = XFinv
            load(b_reg, g.finv_g, {0, 0, 0, 0});
            kittens::zero(mma_reg);
            kittens::mma_AB(mma_reg, accum, b_reg, mma_reg);
            kittens::copy(accum, mma_reg);

            // X = X^T * twinv
            transpose_inplace(accum);
            load(a_reg, g.twinv_g, {0, 0, 0, 0});
            kittens::mul(accum, accum, a_reg);

            // Y = XFinv
            kittens::zero(mma_reg);
            kittens::mma_AB(mma_reg, accum, b_reg, mma_reg);
            kittens::copy(accum, mma_reg);

            // Write Y^T to HBM
            transpose_inplace(accum);
            //for (int st = 0; st < ST; st++) {
            auto accum_st = subtile_inplace<N1>(accum.real, diff % ST);
            store(g.o_real_g, accum_st, {j, i, 0, 0});
            //}
        }
    }
}

// Default size
template<int N> struct fft_template_internal  { using type = fftconv_template<4096, 64>; };
template<> struct fft_template_internal<256> { using type = fftconv_template<4096, 16>; };
template<> struct fft_template_internal<1024> { using type = fftconv_template<4096, 32>; };
template<int N> using fft_template = fft_template_internal<N>::type;

template<int SEQ> typename fft_template<SEQ>::layout::globals setup_templates(
    bf16 *d_u_real, bf16 *d_kf_real, bf16 *d_kf_imag, 
    bf16 *d_f_real, bf16 *d_f_imag, bf16 *d_finv_real, bf16 *d_finv_imag,
    bf16 *d_tw_real, bf16 *d_tw_imag, bf16 *d_twinv_real, bf16 *d_twinv_imag, 
    bf16 *d_o, 
    int B, int H, int N, int N1
) {
    // Select the fft_template based on the value of N
    using fftst = fft_template<SEQ>;
    using globals       = fftst::layout::globals;
    using fft_layout    = fftst::layout::fft_layout;
    using filter_layout = fftst::layout::filter_layout;
    using seq_layout    = fftst::layout::seq_layout;

    // input and output
    seq_layout u_gl{d_u_real, B, H, nullptr, nullptr};
    seq_layout o_gl{d_o, B, H, nullptr, nullptr};

    // filters
    filter_layout kf_gl{
        typename filter_layout::GL{d_kf_real, nullptr, H, nullptr, nullptr},
        typename filter_layout::GL{d_kf_imag, nullptr, H, nullptr, nullptr}
    };
    
    // factors
    fft_layout f_gl{
        typename fft_layout::GL{d_f_real, nullptr, nullptr, nullptr, nullptr},
        typename fft_layout::GL{d_f_imag, nullptr, nullptr, nullptr, nullptr}
    };
    fft_layout tw_gl{
        typename fft_layout::GL{d_tw_real, nullptr, nullptr, nullptr, nullptr},
        typename fft_layout::GL{d_tw_imag, nullptr, nullptr, nullptr, nullptr}
    };
    fft_layout finv_gl{
        typename fft_layout::GL{d_finv_real, nullptr, nullptr, nullptr, nullptr},
        typename fft_layout::GL{d_finv_imag, nullptr, nullptr, nullptr, nullptr}
    };
    fft_layout twinv_t_gl{
        typename fft_layout::GL{d_twinv_real, nullptr, nullptr, nullptr, nullptr},
        typename fft_layout::GL{d_twinv_imag, nullptr, nullptr, nullptr, nullptr}
    };

    globals G{
        o_gl, // O comes first
        u_gl,
        kf_gl,
        f_gl,
        finv_gl,
        tw_gl,
        twinv_t_gl
    };
    return G;
}

template<int SEQ>
void launch(typename fft_template<SEQ>::layout::globals G, int B, int H) {
    using fftst = fft_template<SEQ>;
    // 1 warp
    const dim3 block(NUM_THREADS);
    // Use all 128 SMs
    const dim3 grid(128);
    // Each SM handles b / 128 number of batches
    static int b_tiles = (B + 128 - 1) / 128;
    // Each SM handles all heads
    static int h_tiles = H;


    long mem_size = 1000;

    cudaFuncSetAttribute(
        fftconv_tk<fftst>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        mem_size
    );

    fftconv_tk<fftst><<<grid, block, mem_size>>>(G, b_tiles, h_tiles);
}

#include "harness_async.impl"
