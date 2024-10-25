/* 
Copyright 2024 CGLab, GIST.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. 
*/

#define CUDA_CHECK(val)                                                                                       \
    {                                                                                                         \
        if (val != cudaSuccess)                                                                               \
        {                                                                                                     \
            fprintf(stderr, "Error %s at line %d in file %s\n", cudaGetErrorString(val), __LINE__, __FILE__); \
            exit(1);                                                                                          \
        }                                                                                                     \
    }

#define ZERO4 make_float4(0.f, 0.f, 0.f, 0.f)
#define IMAD(a, b, c) (__mul24((a), (b)) + (c))
#define SCALAR3(a) make_float4((a), (a), (a), 0.f)

#ifndef MAX
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#endif
#ifndef MIN
#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#endif

__forceinline__ __host__ __device__ float4 operator+(const float4 &a, const float4 &b)
{
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

__forceinline__ __host__ __device__ float4 &operator+=(float4 &a, const float4 &b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
    return a;
}

__forceinline__ __host__ __device__ float4 &operator/=(float4 &a, float b)
{
    a.x /= b;
    a.y /= b;
    a.z /= b;
    a.w /= b;
    return a;
}

__forceinline__ __host__ __device__ float4 &operator-=(float4 &a, const float4 &b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    a.w -= b.w;
    return a;
}

__forceinline__ __host__ __device__ float4 &operator*=(float4 &a, float b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
    a.w *= b;
    return a;
}

__forceinline__ __host__ __device__ float4 &operator*=(float4 &a, const float4 &b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
    a.w *= b.w;
    return a;
}

__forceinline__ __host__ __device__ float4 operator-(const float4 &a)
{
    return make_float4(-a.x, -a.y, -a.z, -a.w);
}

__forceinline__ __host__ __device__ float4 operator-(float4 a, float4 b)
{
    return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

__forceinline__ __host__ __device__ float4 operator*(const float4 &a, const float4 &b)
{
    return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}

__forceinline__ __host__ __device__ float4 operator*(const float4 &a, float scalar)
{
    return make_float4(a.x * scalar, a.y * scalar, a.z * scalar, a.w * scalar);
}

__forceinline__ __host__ __device__ float4 operator*(float scalar, const float4 &a)
{
    return a * scalar;
}

__forceinline__ __host__ __device__ float4 operator/(const float4 &a, const float4 &b)
{
    return make_float4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
}

__forceinline__ __host__ __device__ float4 operator/(const float4 &a, float b)
{
    return make_float4(a.x / b, a.y / b, a.z / b, a.w / b);
}

__forceinline__ __device__ float4 fmin3f(float4 a, float4 b) {
    float4 result;
    result.x = fminf(a.x, b.x);
    result.y = fminf(a.y, b.y);
    result.z = fminf(a.z, b.z);
    return result;
}

__forceinline__ __device__ float4 fmax3f(float4 a, float4 b) {
    float4 result;
    result.x = fmaxf(a.x, b.x);
    result.y = fmaxf(a.y, b.y);
    result.z = fmaxf(a.z, b.z);
    return result;
}

__forceinline__ __host__ __device__ bool AllLess3(const float4 &a, const float4 &b)
{
    return a.x < b.x && a.y < b.y && a.z < b.z;
}

__forceinline__ __host__ __device__ bool AllGreater3(const float4 &a, const float4 &b)
{
    return a.x > b.x && a.y > b.y && a.z > b.z;
}

__forceinline__ __host__ __device__ int4 operator<(const float4 &a, const float4 &b)
{
    return make_int4(a.x < b.x, a.y < b.y, a.z < b.z, 0);
}

__forceinline__ __host__ __device__ int4 operator<=(const float4 &a, const float4 &b)
{
    return make_int4(a.x <= b.x, a.y <= b.y, a.z <= b.z, 0);
}

__forceinline__ __host__ __device__ int4 operator<=(float4 a, float b)
{
    return make_int4(a.x <= b, a.y <= b, a.z <= b, a.w <= b);
}

__forceinline__ __host__ __device__ bool Any(const int4 &a)
{
    return a.x || a.y || a.z;
}

__forceinline__ __device__ float Dot(const float4 &a, const float4 &b)
{
    return (a.x * b.x + a.y * b.y + a.z * b.z);
}

__forceinline__ __device__ float norm2(const float4 &a)
{
    return (a.x * a.x + a.y * a.y + a.z * a.z);
}

__forceinline__ __device__ float norm2(const float &a)
{
    return (a * a);
}

__forceinline__ __device__ float Sum(const float4 &a)
{
    return (a.x + a.y + a.z);
}
__forceinline__ __device__ float square(const float &a)
{
    return (a * a);
}

__forceinline__ __host__ __device__ bool all(const int4 &a)
{
    return a.x && a.y && a.z;
}

__forceinline__ __host__ __device__ bool any_negative(float4 a)
{
    return (a.x < 0.f) || (a.y < 0.f) || (a.z < 0.f);
}

__forceinline__ __device__ float4 Sqrtf(const float4 &a)
{
    return make_float4(sqrtf(a.x), sqrtf(a.y), sqrtf(a.z), 0.f);
}

__forceinline__ __device__ float Avg(const float4 &a)
{
    return (a.x + a.y + a.z) / 3.f;
}

__forceinline__ __host__ __device__ float dot(const float4 &a, const float4 &b)
{
    return (a.x * b.x + a.y * b.y + a.z * b.z);
}

__device__ int2 operator<(const int2 &a, const int2 &b)
{
    return make_int2(a.x < b.x, a.y < b.y);
}

__device__ int2 operator>(const int2 &a, const int2 &b)
{
    return make_int2(a.x > b.x, a.y > b.y);
}

__device__ int2 operator+(const int2 &a, const int2 &b)
{
    return make_int2(a.x + b.x, a.y + b.y);
}

__device__ int2 operator-(const int2 &a, const int2 &b)
{
    return make_int2(a.x - b.x, a.y - b.y);
}

__device__ float2 operator*(const float2 &a, const int2 &b)
{
    return make_float2(a.x * b.x, a.y * b.y);
}

__device__ float2 operator+(const float2 &a, const float2 &b)
{
    return make_float2(a.x + b.x, a.y + b.y);
}

__device__ bool any(const int2 &a)
{
    return a.x || a.y;
}

__device__ float2 floor(const float2 a)
{
    return make_float2(floor(a.x), floor(a.y));
}

__device__ int2 float2int(const float2 in)
{
    return make_int2(in.x, in.y);
}

__forceinline__ __host__ __device__ float dot(const float3 &a, const float3 &b)
{
    return (a.x * b.x + a.y * b.y + a.z * b.z);
}

__forceinline__ __host__ __device__ float3 operator-(const float3 &a, const float3 &b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ float distance(const float3 &a, const float3 &b)
{
    return sqrtf(dot((a - b), (a - b)));
}

__host__ __device__ inline int iDivUp(int a, int b)
{
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

__forceinline__ __device__ float4 get3(const float *img, int idx)
{
    return make_float4(img[idx * 3 + 0], img[idx * 3 + 1], img[idx * 3 + 2], 0.f);
}

__device__ float fracf(float x)
{
    return x - floorf(x);
}

__device__ float luminance(const float4 &a)
{
    return a.x * 0.2126f + a.y * 0.7152f + a.z * 0.0722f;
}

__device__ __forceinline__ void printMat(const char *name, const float *m, int rows, int cols)
{
    printf("%s\n", name);
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            printf("%f ", m[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

__device__ __forceinline__ void printVec3(const char *name, float4 *v, int rows)
{
    printf("%s\n", name);
    for (int i = 0; i < rows; i++)
    {
        printf("%f %f %f\n", v[i].x, v[i].y, v[i].z);
    }
    printf("\n");
}

__device__ __forceinline__ void printVec3(const char *name, const float *v, int rows)
{
    printf("%s\n", name);
    for (int i = 0; i < rows; i++)
    {
        printf("%f %f %f\n", v[i * 3 + 0], v[i * 3 + 1], v[i * 3 + 2]);
    }
    printf("\n");
}

__device__ __forceinline__ void MultMatVec(float *out, const float *mat, const float *vec, const int P)
{
    for (int row = 0; row < P; ++row)
    {
        float sum = 0.f;
        for (int col = 0; col < P; ++col)
        {
            sum += mat[row * P + col] * vec[col];
        }
        out[row] = sum;
    }
}

__device__ __forceinline__ void MultVecMat(float *out, const float *vec, const float *mat, const int P)
{
    for (int row = 0; row < P; ++row)
    {
        float sum = 0.f;
        for (int col = 0; col < P; ++col)
        {
            sum += vec[col] * mat[col * P + row];
        }
        out[row] = sum;
    }
}

__device__ __forceinline__ void MultMatVec(float4 *out, const float *mat, const float4 *vec, const int P)
{
    for (int row = 0; row < P; ++row)
    {
        float4 sum = ZERO4;
        for (int col = 0; col < P; ++col)
        {
            sum += mat[row * P + col] * vec[col];
        }
        out[row] = sum;
    }
}

__device__ __forceinline__ void MultVecMat(float4 *out, const float4 *vec, const float *mat, const int P)
{
    for (int row = 0; row < P; ++row)
    {
        float4 sum = ZERO4;
        for (int col = 0; col < P; ++col)
        {
            sum += vec[col] * mat[col * P + row];
        }
        out[row] = sum;
    }
}

__device__ __forceinline__ void MultVecMat(float4 *out, const float4 *vec, const float4 *mat, const int P, bool print = false)
{
    for (int row = 0; row < P; ++row)
    {
        float4 sum = ZERO4;
        for (int col = 0; col < P; ++col)
        {
            sum += vec[col] * mat[col * P + row];
        }
        out[row] = sum;
    }
}