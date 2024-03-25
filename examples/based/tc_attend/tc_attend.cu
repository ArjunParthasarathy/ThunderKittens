#include <iostream>
#include <math.h>
#include <assert.h>
#include <mma.h>
using namespace nvcuda; 

# include <cuda/pipeline>
# include <cooperative_groups.h>
# include "src/kittens.cuh"
# include "src/common/pyutils/torch_helpers.cuh"

#include <ATen/cuda/CUDAContext.h>  // Include necessary for getting the current stream

using namespace kittens;

/*
This function seems to broadcast a single value per worker into shared memory,
then use a single worker to do a reduction on them, and then have
all of the workers take back down that single reduced value.

The way it's written right now looks like a bug due to `shm[warpid] = f;`
^ that looks like a race condition, and only lane 0 should write that.
*/
template<typename op>
__device__ void shm_broadcast(float &f, float *shm, const int workers = 4) {
    auto warpid = threadIdx.x / 32;
    auto lane   = threadIdx.x % 32;
    shm[warpid] = f;
    __syncthreads();
    if(warpid == 0) {
        if(lane == 0) {
            for(auto j = 1; j < workers; j++) {f = op::op(f,shm[j]);}
            for(auto j = 0; j < workers; j++) {shm[j] = f;}
        }
        __syncwarp();
    }
    __syncthreads();
    f = shm[warpid];
}

/*
This function takes in a tile and a row vec and returns a col vec.
First, we multiply every column value by the row vector.
Then, we sum across that axis (and o) and store the result in o.

This could be refactored to be slightly more functional but that's alright.
*/
// GEMV
__device__ void gemv(rt_fl_1x4<>::col_vec  &o, rt_fl_1x4<>::row_vec &x, rt_fl_1x4<> &a) { 
    rt_fl_1x4<> t;
    // The accumulator is row x column; row multiply means that each row is multiplied by a column matrix. 
    mul_col(t, a, x); // multiply vv in place with aa: a * v.unsqueeze(1) // row, row, col
    row_sum(o, t, o); // aa.sum(0) sum across all the rows 
}

/*
This function seems to be the transpose of the above function.
*/
// GEMV
__device__void gemv_two(rt_fl_4x1<>::row_vec  &o, rt_fl_4x1<>::col_vec &x, rt_fl_4x1<> &a) { 
    rt_fl_4x1<> t;
    copy(t, a);
    // The accumulator is row x column; row multiply means that each row is multiplied by a column matrix. 
    // mul_row(t, a, x); // SA: uncommenting this line leads to nans in the output
    col_sum(o, t, o); // aa.sum(0) sum across all the rows 
}


/*
This is either global to reg or shared to reg, can't quite tell yet.

Either way I'm going to find a way to rename it to "load".
*/
static void __device__
vec_to_rvec(rt_fl_4x1<>::col_vec &dst, const bf16 *src) {
    using T = bf16;
    using U = float;
    auto row = kittens::laneid() / 4;
    __syncwarp();    
    for(auto h = 0; h < dst.outer_dim; h++) {
        dst[h][0].x = base_types::convertor<U, T>::convert(src[h*kittens::TILE_DIM + row]);    
        dst[h][1].x = base_types::convertor<U, T>::convert(src[h*kittens::TILE_DIM + row + 8]); 
    }
}

/*
This is the equivalent "store" of above.
*/
static void __device__
rvec_to_vec(bf16 *dst, rt_fl_1x4<>::col_vec &src) {
    using U = bf16;
    using T = float;
    auto row = kittens::laneid() / 4;
    __syncwarp();
    if(kittens::laneid() % 4 == 0) { // only the leaders write
        for(auto h = 0; h < src.outer_dim; h++) {
            dst[h*TILE_DIM + row]     = base_types::convertor<U, T>::convert(src[h][0].x);  
            dst[h*TILE_DIM + row + 8] = base_types::convertor<U, T>::convert(src[h][1].x);
        }
    }    
}


/*
Main kernel.
*/
template<typename H, typename T>
__global__
void sliding_window_ker_hack(int n, int j, bool just_q, const T* __q, const T* __k, const T* __v, T* __o) {
    
    auto warpid = kittens::warpid();
    const int d = 64;
    const int window_size = 64;
    constexpr int workers = 4;
    const int threads = workers * kittens::WARP_SIZE;
    auto head_offset  = blockIdx.x * n * d;
    using block = kittens::block<workers>;
    
    const H* _q = device_cast(__q) + blockIdx.x*d;
    const H* _k = device_cast(__k) + head_offset;
    const H* _v = device_cast(__v) + head_offset;
          H* _o = device_cast(__o) + blockIdx.x*d; // just a single vector

    // Register
    rt_fl_1x4<> k_slice; 
    rt_fl_4x1<> v_slice; // Each of the 4 workers stores a column
    rt_fl_1x4<>::row_vec qv; // full local copy 
    rt_fl_1x4<>::col_vec ws; 
    rt_fl_4x1<>::col_vec wv; // full local copy 
    rt_fl_4x1<>::row_vec os; // shards
    auto vec_idx = 0;
    __syncthreads();
    load(qv, _q + vec_idx); // every warp gets a full copy of q; These are column slices of the matrix.: | K_1 | K_2 | K_3 |

    // Shared
    extern __shared__ alignment_dummy __shm[]; // this is the CUDA shared memory
    shared_allocator al = shared_allocator::create_allocator((int*)&__shm[0]);
    st_bf_1x4<ducks::st_layout::xor_swizzle>::row_vec &w = al.allocate<st_bf_1x4<ducks::st_layout::xor_swizzle>::row_vec>();
    __shared__ float _max[workers], _sum[workers];  

    // Option A (References / Following the tests)
    const auto start_idx = 0;
    st_bf_4x4<ducks::st_layout::xor_swizzle> &k = al.allocate<st_bf_4x4<ducks::st_layout::xor_swizzle>>(); // We use 4x4 since 4x16 is 64 window size
    st_bf_4x4<ducks::st_layout::xor_swizzle> &v = al.allocate<st_bf_4x4<ducks::st_layout::xor_swizzle>>();
    block::load(k, _k + start_idx, d); // One warp loads from global to shared
    block::load(v, _v + start_idx, d);
    __syncthreads();
    auto subtile = subtile_inplace<1,4>(k, warpid, 0); // All the other warps load from shared to shared
    load(k_slice, subtile);
    __syncthreads();


    one(k_slice);
    one(v_slice);


    // The algorithm.
    // qs = [q for j in range(4)] # broadcast q to each warp
    // ks = [k[:,j*d//4:(j+1)*d//4] for j in range(4)] # shard k
    // ws = [torch.einsum("d, de->e", qs[j],ks[j]) for j in range(4)]
    zero(ws);
    gemv(ws, qv, k_slice);

    // local_max = [ws[j].max() for j in range(4)] # compute local, then global max
    // the_max = torch.tensor(local_max).max()
    float local_max= -INFINITY;
    max(ws, ws, local_max);
    shm_broadcast<base_ops::mul>(local_max, _max);
    
    // ews = [torch.exp(ws[j] - the_max) for j in range(4)]
    sub(ws, ws, local_max);
    exp(ws, ws);

    // es  = [ews[j].sum() for j in range(4)]
    float local_sum = 0.f;
    add(ws, ws, local_sum);
    shm_broadcast<base_ops::sum>(local_sum, _sum);
    
    // w  /= the_sum
    div(ws, ws, local_sum);

    // broadcast w back to shared memory
    rvec_to_vec(&w.data[warpid*kittens::TILE_DIM], ws);
    __syncthreads(); // let the writes complete
    vec_to_rvec(wv, w.data); // read the *whole* v here.
    
    // we want a column stripe of V
    auto subtile_v = subtile_inplace<4,1>(v, 0, warpid);
    // load(v_slice, subtile_v); // SA: Uncommenting this leads to static asserts in the output (even if i uncomment the thread_block_loads)
    zero(os);
    gemv_two(os, wv, v_slice);
    
    // now we have a fragment of v and we write, this write is to *global* memory.
    store(_o + warpid*kittens::TILE_DIM, os);
}

void 
sliding_window(int j,   
    torch::Tensor q, torch::Tensor k, torch::Tensor v, 
    torch::Tensor o) {

    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);
    CHECK_INPUT(o);
    
    uint batch = q.size(0);
    uint head  = q.size(1);
    uint d     = q.size(3);
    TORCH_CHECK(d == 64, "Only dimension 64 implemented...");

    bool k_same = true, v_same = true;
    for(auto i = 0; i < 2; i++) { 
        k_same &= q.size(i) == k.size(i);
        v_same &= q.size(i) == v.size(i);
    }
    k_same &= d == k.size(3);
    v_same &= d == v.size(3);
    uint n     = k.size(2);
    v_same &= v.size(2) == n;

    // This is just a restriction of what we're doing now...
    TORCH_CHECK(k_same, "X and K_out should be same size");
    TORCH_CHECK(v_same, "X and V_out should be same size");
    
    const int workers = 4;
    using H = bf16;
    using T = c10::BFloat16;

    int threads = workers * kittens::WARP_SIZE;

    auto stream_wrapper = at::cuda::getCurrentCUDAStream(q.device().index());
    cudaStream_t stream = stream_wrapper.stream();
    sliding_window_ker_hack<H,T><<<batch*head,threads,0,stream>>>(n, j, q.size(2) == 1,
                        q.data_ptr<T>(), k.data_ptr<T>(), v.data_ptr<T>(), 
                        o.data_ptr<T>());
}

