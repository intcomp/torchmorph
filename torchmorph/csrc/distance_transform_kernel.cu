#include <cuda_runtime.h>
#include <cmath>

__global__ void edt_pass1_rows(const float* input, float* temp, int H, int W) {
    int y = blockIdx.x;
    if (y >= H) return;

    extern __shared__ float sdata[];
    float* f = sdata;
    int* v = (int*)(sdata + W);
    float* z = (float*)(v + W + 1);

    for (int x = threadIdx.x; x < W; x += blockDim.x) {
        float val = input[y * W + x];
        // 【关键任务修改】
        // 如果像素是 0 (背景)，则为源点 (距离0)；否则为无穷远。
        f[x] = (val < 0.5f) ? 0.0f : 1e10f;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        int k = 0;
        v[0] = 0;
        z[0] = -1e10f;
        z[1] = 1e10f;

        for (int q = 1; q < W; q++) {
            float s;
            while (true) {
                int p = v[k];
                s = ((f[q] + q * q) - (f[p] + p * p)) / (2.0f * (q - p));
                if (s > z[k]) { break; }
                if (k == 0) { break; }
                k--;
            }
            k++;
            v[k] = q;
            z[k] = s;
            z[k + 1] = 1e10f;
        }

        k = 0;
        for (int q = 0; q < W; q++) {
            while (z[k + 1] < q) k++;
            int p = v[k];
            temp[y * W + q] = (q - p) * (q - p) + f[p];
        }
    }
}
// PASS 2: 对每一列进行操作
__global__ void edt_pass2_cols(const float* temp, float* output, int H, int W) {
    int x = blockIdx.x;
    if (x >= W) return;

    extern __shared__ float sdata[];
    float* f = sdata;
    int* v = (int*)(sdata + H);
    float* z = (float*)(v + H + 1);

    for (int y = threadIdx.x; y < H; y += blockDim.x) {
        f[y] = temp[y * W + x];
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        int k = 0;
        v[0] = 0;
        z[0] = -1e10f;
        z[1] = 1e10f;

        for (int q = 1; q < H; q++) {

            float s;
            while (true) {
                int p = v[k];
                s = ((f[q] + q * q) - (f[p] + p * p)) / (2.0f * (q - p));
                if (s > z[k]) {
                    break;
                }
                if (k == 0) {
                    break;
                }
                k--;
            }
            k++;
            v[k] = q;
            z[k] = s;
            z[k + 1] = 1e10f;
        }

        k = 0;
        for (int q = 0; q < H; q++) {
            while (z[k + 1] < q) k++;
            int p = v[k];
            output[q * W + x] = sqrtf((q - p) * (q - p) + f[p]);
        }
    }
}