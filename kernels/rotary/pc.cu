#include "kittens.cuh"
#include "prototype.cuh"

using namespace kittens;
using namespace kittens::prototype;
template<int _headdim, int _warps> struct rotary_layout {
    static constexpr int headdim = _headdim, warps = _warps;
    using seq_tile    = st_bf<16, headdim>;
    using seq_global  = gl<bf16, -1, -1, -1, headdim, seq_tile>;
    using rope_global = gl<bf16,  1,  1, -1, headdim/2>;
    struct globals {
        seq_global o, x;
        rope_global sin, cos;
        int batches; // how many batches per block, for sizing grid
    };
    struct input_block    { seq_tile x[warps]; };
    struct output_block   { seq_tile o[warps]; };
    struct producer_state { int active_warps;  };
    struct consumer_state { rt_fl<16, headdim/2> sin, cos; }; // long-resident tiles
};
template<int _headdim> struct rotary_template {
    static constexpr int headdim=_headdim, NUM_CONSUMER_WARPS=8, NUM_BLOCKS=1, OUTPUT_PIPE_STAGES=3, INPUT_PIPE_STAGES=3;
    using layout = rotary_layout<headdim, NUM_CONSUMER_WARPS>;
    __device__ static inline int iters(const typename layout::globals &g) {
        return min(g.batches, (int)(g.x.batch-blockIdx.y*g.batches)) * g.x.depth; // batches*heads handled by block
    }
    struct producer {
        __device__ static void setup(producer_setup_args<layout> args) {
            warpgroup::producer_registers();
            args.state.active_warps = min((int)NUM_CONSUMER_WARPS,
                                          (int)(args.globals.x.rows/16 - blockIdx.x*NUM_CONSUMER_WARPS));
        }
        __device__ static void load(producer_load_args<layout> args) {
            if(warpgroup::warpid() == args.iter%4) {
                kittens::coord idx = { blockIdx.y*args.globals.batches+args.iter/args.globals.x.depth,
                                       args.iter%args.globals.x.depth,
                                       blockIdx.x*NUM_CONSUMER_WARPS,
                                       0 };
                tma::expect_bytes(args.inputs_arrived, sizeof(layout::seq_tile)*args.state.active_warps);
                for(int i = 0; i < args.state.active_warps; i++) {
                    tma::load_async(args.input.x[i], args.globals.x, {idx.b,idx.d,idx.r+i,idx.c}, args.inputs_arrived);
                }
                arrive(args.inputs_arrived, 3);
            }
        }
        __device__ static void store(producer_store_args<layout> args) {
            if(warpgroup::warpid() == args.iter%4) {
                kittens::coord idx = { blockIdx.y*args.globals.batches+args.iter/args.globals.x.depth,
                                       args.iter%args.globals.x.depth,
                                       blockIdx.x*NUM_CONSUMER_WARPS,
                                       0 };
                for(int i = 0; i < args.state.active_warps; i++) {
                    tma::store_async(args.globals.o, args.output.o[i], {idx.b,idx.d,idx.r+i,idx.c});
                }
                tma::store_async_read_wait();
                arrive(args.outputs_finished, 4);
            }
        }
    };
    struct consumer {
        __device__ static void setup(consumer_setup_args<layout> args) {
            warpgroup::consumer_registers<NUM_CONSUMER_WARPS/4>();
            kittens::coord idx = { blockIdx.x*NUM_CONSUMER_WARPS + warpid(), 0 };
            load(args.state.sin, args.globals.sin, idx); // could be better coalesced but doing just once
            load(args.state.cos, args.globals.cos, idx);
        }
        __device__ static void work(consumer_work_args<layout> args) {
            rt_fl<16, headdim> x;
            rt_fl<16, headdim/2> x1, x2, temp1, temp2;
            load(x, args.input.x[warpid()]);
            arrive(args.inputs_finished);
            for(int i = 0; i < headdim/32; i++) {
                #pragma unroll
                for(int j = 0; j < 4; j++) {
                    x1.tiles[0][i].data[j] = x.tiles[0][i].data[j];
                    x2.tiles[0][i].data[j] = x.tiles[0][i+headdim/32].data[j];
                }
            }
            mul(temp1, x1, args.state.cos);
            mul(temp2, x2, args.state.cos);
            mul(x2, x2, -1.f);
            mul(x1, x1, args.state.sin);
            mul(x2, x2, args.state.sin);
            add(temp1, temp1, x2);
            add(temp2, temp2, x1);
            for(int i = 0; i < headdim/32; i++) {
                #pragma unroll
                for(int j = 0; j < 4; j++) {
                    x.tiles[0][i].data[j]            = temp1.tiles[0][i].data[j];
                    x.tiles[0][i+headdim/32].data[j] = temp2.tiles[0][i].data[j];
                }
            }
            store(args.output.o[warpid()], x);
            __syncwarp();
            arrive(args.outputs_arrived);
        }
    };
};

#include <iostream>
#include <string>
#include <fstream>

#define ATTN_B 32
#define ATTN_N 2048
#define ATTN_H 16 // launches
#define ATTN_D 128 // make sure to change in the kernel rotary.cu as well
using rope_t = rotary_template<ATTN_D>;

const int ATTN_D_2 = ATTN_D / 2;

#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )
inline void __cudaCheckError( const char *file, const int line ) {
    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
    // More careful checking. However, this will affect performance.
    // Comment away if needed.
    err = cudaDeviceSynchronize();
    if( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
                file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
}

// Function to calculate the number of floating-point operations
long long flops(int batch, int seqlen, int headdim, int nheads) {
    // feature map
    long long f = batch * static_cast<long long>(seqlen) * nheads * (headdim / 2);
    f += batch * static_cast<long long>(seqlen) * nheads * (headdim / 2);
    f += batch * static_cast<long long>(seqlen) * nheads * (headdim / 2);
    return f*2; // for q and k
}

// Function to calculate the efficiency in teraflops
double efficiency(long long flop, double time) {
    // Convert flop to teraflops and time to milliseconds
    double tflops = flop / 1e12;
    double time_ms = time / 1e6;
    return tflops / time_ms;
}

int main(int argc, char **argv) {
    std::cout << "Entered main!" << std::endl;

    // create dummy variables that are the right size
    constexpr int TOTAL_ELEMENTS_X = ATTN_B*ATTN_H*ATTN_N*ATTN_D;
    constexpr int TOTAL_ELEMENTS_O = ATTN_B*ATTN_H*ATTN_N*ATTN_D;
    constexpr int TOTAL_UNIQUE_ELEMENTS_X = ATTN_N*ATTN_D;
    constexpr int TOTAL_UNIQUE_ELEMENTS_O = ATTN_N*ATTN_D;

    float *x = new float[TOTAL_UNIQUE_ELEMENTS_X];
    bf16 *x_bf = new bf16[TOTAL_ELEMENTS_X];

    float *o_ref = new float[TOTAL_UNIQUE_ELEMENTS_O];
    bf16 *o_bf = new bf16[TOTAL_ELEMENTS_O];
    float *o = new float[TOTAL_ELEMENTS_O];

    constexpr int TOTAL_ELEMENTS_COS_IN = ATTN_N*ATTN_D_2;
    float *cos_in = new float[TOTAL_ELEMENTS_COS_IN];
    bf16  *cos_in_bf = new bf16[TOTAL_ELEMENTS_COS_IN];
    float *sin_in = new float[TOTAL_ELEMENTS_COS_IN];
    bf16  *sin_in_bf = new bf16[TOTAL_ELEMENTS_COS_IN];

    // set the inputs
    if(argc > 1) {
        std::ifstream infile(argv[1]);
        printf("Loading from %s\n", argv[1]);
        std::cout << "Starting to enter!" << std::endl;
        for(int i = 0; i < TOTAL_UNIQUE_ELEMENTS_X; i++) {  
            infile >> x[i];   
            // if (i < 5)  { std::cout << x[i] << std::endl; } 
        }
        std::cout << "Finished loading X" << std::endl;
        for(int i = 0; i < TOTAL_UNIQUE_ELEMENTS_O; i++) {  
            infile >> o_ref[i];  
            // o_ref[i] = x[i];
            // if (i < 5)  { std::cout << o_ref[i] << std::endl; } 
        }
        std::cout << "Finished loading O_REF" << std::endl;

        for(int i = 0; i < TOTAL_ELEMENTS_COS_IN; i++) { 
            infile >> cos_in[i];   
            // if (i < 5)  { std::cout << cos_in[i] << std::endl; } 
        }
        std::cout << "Finished loading COS_IN" << std::endl;
        for(int i = 0; i < TOTAL_ELEMENTS_COS_IN; i++) {  
            infile >> sin_in[i];   
            // if (i < 50)  { std::cout << sin_in[i] << std::endl; } 
        }
        std::cout << "Finished loading SIN_IN" << std::endl;

        std::cout << "Finished loading file from " << argv[1] << "!" << std::endl;
    }

    // set the inputs
    for(int i = 0; i < TOTAL_ELEMENTS_X; i++) {
        x_bf[i] = __float2bfloat16(x[i % TOTAL_UNIQUE_ELEMENTS_X]);
        // if (i < 5) { std::cout << x[i % TOTAL_UNIQUE_ELEMENTS_X] << std::endl; } 
    }
    for(int i = 0; i < TOTAL_ELEMENTS_COS_IN; i++) {
        cos_in_bf[i] = __float2bfloat16(cos_in[i % TOTAL_ELEMENTS_COS_IN]);
        // if (i < 5) { std::cout << __bf/loat162float(cos_in_bf[i]) << std::endl; } 
        sin_in_bf[i] = __float2bfloat16(sin_in[i % TOTAL_ELEMENTS_COS_IN]);
    }

    bf16 *d_x, *d_o;
    cudaMalloc(&d_x, TOTAL_ELEMENTS_X * sizeof(bf16));
    cudaMalloc(&d_o, TOTAL_ELEMENTS_O * sizeof(bf16));
    cudaMemcpy(d_x, x_bf, TOTAL_ELEMENTS_X * sizeof(bf16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_o, o_bf, TOTAL_ELEMENTS_O * sizeof(bf16), cudaMemcpyHostToDevice);

    bf16 *d_cos_in, *d_sin_in;
    cudaMalloc(&d_cos_in, TOTAL_ELEMENTS_COS_IN * sizeof(bf16));
    cudaMalloc(&d_sin_in, TOTAL_ELEMENTS_COS_IN * sizeof(bf16));
    cudaMemcpy(d_cos_in, cos_in_bf, TOTAL_ELEMENTS_COS_IN * sizeof(bf16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sin_in, sin_in_bf, TOTAL_ELEMENTS_COS_IN * sizeof(bf16), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    CudaCheckError();

    unsigned long mem_size = (MAX_SHARED_MEMORY-2048);
    std::cout << "Setting max block shared memory to " << mem_size << std::endl;
    using T = kittens::bf16;
    using H = kittens::bf16;
    cudaFuncSetAttribute(
        pc<rope_t>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        mem_size
    );

    constexpr int BATCHES_PER_BLOCK = 4;
    rope_t::layout::seq_global Og{d_o, ATTN_B, ATTN_H, ATTN_N, nullptr};
    rope_t::layout::seq_global Xg{d_x, ATTN_B, ATTN_H, ATTN_N, nullptr};
    rope_t::layout::rope_global SINg{d_sin_in, nullptr, nullptr, ATTN_N, nullptr};
    rope_t::layout::rope_global COSg{d_cos_in, nullptr, nullptr, ATTN_N, nullptr};
    rope_t::layout::globals g{Og, Xg, SINg, COSg, BATCHES_PER_BLOCK};

    constexpr int ROWS_PER_BLOCK = rope_t::NUM_CONSUMER_WARPS * rope_t::layout::seq_tile::rows;
    dim3 grid((ATTN_N+ROWS_PER_BLOCK-1)/ROWS_PER_BLOCK, (ATTN_B+BATCHES_PER_BLOCK-1)/BATCHES_PER_BLOCK);
    dim3 block(num_threads<rope_t>);

    const int ITER = 1;
    cudaDeviceSynchronize();
    CudaCheckError();
    std::cout << "Starting kernel with grid of size " << grid.x << ", " << grid.y << "\n";
    const auto start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < ITER; i++) {  
        pc<rope_t><<<grid, block, mem_size>>>(g);  
    }
    cudaDeviceSynchronize();
    const auto finish = std::chrono::high_resolution_clock::now();
    CudaCheckError();
    std::cout << "Finished kernel\n";
    
    // check correctness
    cudaMemcpy(o_bf, d_o, TOTAL_ELEMENTS_O * sizeof(bf16), cudaMemcpyDeviceToHost);
    for(int i = 0; i < TOTAL_ELEMENTS_O; i++) { o[i] = __bfloat162float(o_bf[i]); }
    bool good = true;
    std::ofstream o_ref_file("printouts/o_ref.txt");
    std::ofstream o_file("printouts/o.txt");
    std::ofstream diff_file("printouts/diff.txt");
    std::cout << "Total elements: " << TOTAL_ELEMENTS_O << std::endl;
    std::cout << "Total unique elements: " << TOTAL_UNIQUE_ELEMENTS_O << std::endl;
    for(int i = 0; i < TOTAL_ELEMENTS_O; i++) {
        float diff = o[i] - o_ref[i % TOTAL_UNIQUE_ELEMENTS_O];
        if(i < TOTAL_UNIQUE_ELEMENTS_O) {
            o_ref_file << o_ref[i % TOTAL_UNIQUE_ELEMENTS_O] << ' ';
            o_file << o[i] << ' ';
            diff_file << diff << ' ';
        }
        if(abs(diff) > 0.1 || isnan(diff)) {
            good = false;
        }
    }
    if(good) std::cout << "Correct out :)\n";
    else std::cout << "Incorrect out :(\n";
    std::cout << "Average execution time for " << ITER << " ITERS (Q and K): " << std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count() << " us" << std::endl;

    // calculate efficiency
    long long f = flops(ATTN_B, ATTN_N, ATTN_D, ATTN_H);
    double e = efficiency(f, std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count());
    std::cout << "FLOPS: " << f << std::endl;
    std::cout << "Efficiency: " << e << " TFLOPS" << std::endl;

    cudaFree(d_x);
    cudaFree(d_sin_in);
    cudaFree(d_cos_in);
    cudaFree(d_o);

    delete[] x, o, o_ref, x_bf, o_bf;
    delete[] sin_in, sin_in_bf, cos_in, cos_in_bf;
    return 0;
}