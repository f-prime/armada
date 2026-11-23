/* GGUF Q8_0 matrix-vector product: FP16 scale and 32 signed bytes per block.
 * Keep float activations; no temporary decoded weight matrix or requantization. */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#define HAVE_AVX2 1
#include <immintrin.h>
#elif defined(__aarch64__) && defined(__ARM_NEON)
#define HAVE_NEON 1
#include <arm_neon.h>
#endif

static float scales[65536];
static inline float scale(const uint8_t *p) { return scales[p[0] | ((unsigned)p[1] << 8)]; }

typedef float (*dot_fn)(const uint8_t *, const float *, Py_ssize_t);
static float scalar(const uint8_t *p, const float *x, Py_ssize_t n) {
    float y = 0;
    for (; n; --n, p += 34, x += 32) {
        float sum = 0;
        for (int j = 0; j < 32; ++j) sum += x[j] * (int8_t)p[2+j];
        y += scale(p) * sum;
    }
    return y;
}
static dot_fn dot = scalar;
typedef void (*dot4_fn)(const uint8_t *, const float *, Py_ssize_t, float *);
static dot4_fn dot4;

#ifdef HAVE_AVX2
__attribute__((target("avx2")))
static float avx2(const uint8_t *p, const float *x, Py_ssize_t n) {
    __m256 y = _mm256_setzero_ps();
    for (; n; --n, p += 34, x += 32) {
        __m256 sum = _mm256_setzero_ps();
        for (int j = 0; j < 32; j += 8) {
            __m128i bytes = _mm_loadl_epi64((const __m128i *)(p+2+j));
            __m256 w = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(bytes));
            sum = _mm256_add_ps(sum, _mm256_mul_ps(w, _mm256_loadu_ps(x+j)));
        }
        y = _mm256_add_ps(y, _mm256_mul_ps(_mm256_set1_ps(scale(p)), sum));
    }
    float lanes[8];
    _mm256_storeu_ps(lanes, y);
    float result = 0;
    for (int i = 0; i < 8; ++i) result += lanes[i];
    return result;
}
__attribute__((target("avx2")))
static void avx2_four(const uint8_t *p, const float *x, Py_ssize_t n, float *out) {
    const Py_ssize_t stride = n*32;
    __m256 y[4] = {0};
    for (; n; --n, p += 34, x += 32) {
        __m256 sum[4] = {0};
        for (int j = 0; j < 32; j += 8) {
            __m128i bytes = _mm_loadl_epi64((const __m128i *)(p+2+j));
            __m256 w = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(bytes));
            for (int t = 0; t < 4; ++t)
                sum[t] = _mm256_add_ps(sum[t], _mm256_mul_ps(w, _mm256_loadu_ps(x+t*stride+j)));
        }
        __m256 s = _mm256_set1_ps(scale(p));
        for (int t = 0; t < 4; ++t) y[t] = _mm256_add_ps(y[t], _mm256_mul_ps(s, sum[t]));
    }
    for (int t = 0; t < 4; ++t) {
        float lanes[8], result = 0;
        _mm256_storeu_ps(lanes, y[t]);
        for (int i = 0; i < 8; ++i) result += lanes[i];
        out[t] = result;
    }
}
#elif defined(HAVE_NEON)
static float neon(const uint8_t *p, const float *x, Py_ssize_t n) {
    float32x4_t y = vdupq_n_f32(0);
    for (; n; --n, p += 34, x += 32) {
        float32x4_t sum = vdupq_n_f32(0);
        for (int j = 0; j < 32; j += 8) {
            int16x8_t w = vmovl_s8(vld1_s8((const int8_t *)(p+2+j)));
            sum = vfmaq_f32(sum, vcvtq_f32_s32(vmovl_s16(vget_low_s16(w))), vld1q_f32(x+j));
            sum = vfmaq_f32(sum, vcvtq_f32_s32(vmovl_s16(vget_high_s16(w))), vld1q_f32(x+j+4));
        }
        y = vfmaq_n_f32(y, sum, scale(p));
    }
    return vaddvq_f32(y);
}
static void neon_four(const uint8_t *p, const float *x, Py_ssize_t n, float *out) {
    const Py_ssize_t stride = n*32;
    float32x4_t y[4] = {0};
    for (; n; --n, p += 34, x += 32) {
        float32x4_t sum[4] = {0};
        for (int j = 0; j < 32; j += 8) {
            int16x8_t w = vmovl_s8(vld1_s8((const int8_t *)(p+2+j)));
            float32x4_t lo = vcvtq_f32_s32(vmovl_s16(vget_low_s16(w)));
            float32x4_t hi = vcvtq_f32_s32(vmovl_s16(vget_high_s16(w)));
            for (int t = 0; t < 4; ++t) {
                sum[t] = vfmaq_f32(sum[t], lo, vld1q_f32(x+t*stride+j));
                sum[t] = vfmaq_f32(sum[t], hi, vld1q_f32(x+t*stride+j+4));
            }
        }
        float s = scale(p);
        for (int t = 0; t < 4; ++t) y[t] = vfmaq_n_f32(y[t], sum[t], s);
    }
    for (int t = 0; t < 4; ++t) out[t] = vaddvq_f32(y[t]);
}
#endif

static PyObject *matvec(PyObject *self, PyObject *args) {
    Py_buffer w, x, out;
    int batch = 1;
    if (!PyArg_ParseTuple(args, "y*y*w*|i", &w, &x, &out, &batch)) return NULL;
    Py_ssize_t blocks = 0, rows = 0;
    if (batch >= 1 && batch <= 32) {
        blocks = x.len / (128 * batch);
        rows = out.len / (4 * batch);
    }
    int valid = blocks > 0 && rows > 0 && x.len % (128 * batch) == 0 && out.len % (4 * batch) == 0
        && w.len % rows == 0 && w.len / rows == blocks * 34
        && (uintptr_t)x.buf % _Alignof(float) == 0 && (uintptr_t)out.buf % _Alignof(float) == 0;
    if (valid) {
        Py_BEGIN_ALLOW_THREADS
        if (batch == 1) {
            for (Py_ssize_t i = 0; i < rows; ++i)
                ((float *)out.buf)[i] = dot((const uint8_t *)w.buf + i*blocks*34, x.buf, blocks);
        } else {
            /* Consume a packed row for every prompt position while it is hot.
             * Row-major [output feature, position] allows disjoint worker slices. */
            for (Py_ssize_t i = 0; i < rows; ++i) {
                int t = 0;
                if (dot4)
                    for (; t+4 <= batch; t += 4)
                        dot4((const uint8_t *)w.buf + i*blocks*34, (const float *)x.buf + t*(blocks*32),
                             blocks, (float *)out.buf + i*batch+t);
                for (; t < batch; ++t)
                    ((float *)out.buf)[i*batch+t] = dot((const uint8_t *)w.buf + i*blocks*34,
                        (const float *)x.buf + t*(blocks*32), blocks);
            }
        }
        Py_END_ALLOW_THREADS
    }
    PyBuffer_Release(&out);
    PyBuffer_Release(&x);
    PyBuffer_Release(&w);
    if (!valid) {
        PyErr_SetString(PyExc_ValueError, "invalid Q8_0 matvec buffers");
        return NULL;
    }
    Py_RETURN_NONE;
}

/* Decode only one 32-weight group at a time, reused across all prompt
 * positions. Quants stay packed in RAM; activations remain float32. */
static inline void unpack_group(float *v, const uint8_t *p, int type, int g) {
    if (type == 2 || type == 3) { /* Q4_0, Q4_1 */
        float d = scale(p), m = type == 2 ? -8*d : scale(p+2);
        const uint8_t *q = p + (type == 2 ? 2 : 4);
        for (int j = 0; j < 16; ++j) {
            v[j] = d*(q[j]&15) + m;
            v[j+16] = d*(q[j]>>4) + m;
        }
    } else if (type == 14) { /* Q6_K: signed scale for each 16 weights */
        const uint8_t *lo = p + (g/4)*64 + (g%2)*32;
        const uint8_t *hi = p + 128 + (g/4)*32;
        for (int half = 0; half < 2; ++half) {
            float d = scale(p+208) * (int8_t)p[192+2*g+half];
            for (int j = 16*half; j < 16*half+16; ++j) {
                int q = ((lo[j] >> (4*((g/2)%2))) & 15) | (((hi[j] >> (2*(g%4))) & 3)<<4);
                v[j] = d*(q-32);
            }
        }
    } else { /* Q4_K, Q5_K: two packed 6-bit scale/min fields per group */
        const uint8_t *s = p+4;
        int sc = g < 4 ? s[g]&63 : (s[4+g]&15) | ((s[g-4]>>6)<<4);
        int mn = g < 4 ? s[g+4]&63 : (s[4+g]>>4) | ((s[g]>>6)<<4);
        float d = scale(p)*sc, m = scale(p+2)*mn;
        const uint8_t *lo = p + (type == 13 ? 48 : 16) + (g/2)*32;
        for (int j = 0; j < 32; ++j) {
            int q = (lo[j] >> (4*(g%2))) & 15;
            if (type == 13) q |= (((p[16+j] >> g)&1)<<4);
            v[j] = d*q - m;
        }
    }
}

typedef void (*group_fn)(const float *, const float *, Py_ssize_t, int, float *);
static void group_scalar(const float *v, const float *x, Py_ssize_t stride, int batch, float *out) {
    for (int t = 0; t < batch; ++t, x += stride) {
        float sum = 0;
        for (int j = 0; j < 32; ++j) sum += v[j]*x[j];
        out[t] += sum;
    }
}
static group_fn accumulate = group_scalar;
typedef void (*unpack_fn)(float *, const uint8_t *, int, int);
static unpack_fn unpack = unpack_group;
typedef float (*lowbit_row_fn)(const uint8_t *, const float *, Py_ssize_t, int, int, int);
static lowbit_row_fn lowbit_row;
/* Locate one group's bit planes and affine parameters for SIMD unpacking. */
typedef struct {
    const uint8_t *lo, *hi;
    int shift, hshift, hmask;
    float d, m;
} group_layout;
static inline group_layout layout(const uint8_t *p, int type, int g) {
    group_layout a = {0};
    if (type < 12) {
        a.d = scale(p); a.m = type == 2 ? -8*a.d : scale(p+2);
        a.lo = p + (type == 2 ? 2 : 4);
    } else if (type == 14) {
        a.lo = p+(g/4)*64+(g%2)*32; a.hi = p+128+(g/4)*32;
        a.shift = 4*((g/2)%2); a.hshift = 2*(g%4); a.hmask = 3;
        a.d = scale(p+208)*(int8_t)p[192+2*g]; a.m = -32*a.d;
    } else {
        const uint8_t *s = p+4;
        a.d = scale(p)*(g < 4 ? s[g]&63 : (s[4+g]&15) | ((s[g-4]>>6)<<4));
        a.m = -scale(p+2)*(g < 4 ? s[g+4]&63 : (s[4+g]>>4) | ((s[g]>>6)<<4));
        a.shift = 4*(g%2); a.lo = p+(type == 13 ? 48 : 16)+(g/2)*32;
        if (type == 13) { a.hi = p+16; a.hshift = g; a.hmask = 1; }
    }
    return a;
}
#ifdef HAVE_AVX2
__attribute__((target("avx2")))
static void unpack_avx2(float *v, const uint8_t *p, int type, int g) {
    group_layout a = layout(p,type,g);
    float d = a.d, m = a.m;
    for (int j=0; j<32; j+=8) {
        if (j == 16 && type == 14) { d = scale(p+208)*(int8_t)p[193+2*g]; m = -32*d; }
        int off = type < 12 ? j%16 : j;
        __m128i b = _mm_loadl_epi64((const __m128i *)(a.lo+off));
        b = _mm_and_si128(_mm_srl_epi16(b,_mm_cvtsi32_si128(type < 12 && j>=16 ? 4 : a.shift)),_mm_set1_epi8(15));
        if (a.hi) {
            __m128i h = _mm_loadl_epi64((const __m128i *)(a.hi+j));
            h = _mm_and_si128(_mm_srl_epi16(h,_mm_cvtsi32_si128(a.hshift)),_mm_set1_epi8((char)a.hmask));
            b = _mm_or_si128(b,_mm_slli_epi16(h,4));
        }
        __m256 f = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(b));
        _mm256_storeu_ps(v+j,_mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(d),f),_mm256_set1_ps(m)));
    }
}
__attribute__((target("avx2")))
static void group_avx2(const float *v, const float *x, Py_ssize_t stride, int batch, float *out) {
    __m256 a = _mm256_loadu_ps(v), b = _mm256_loadu_ps(v+8);
    __m256 c = _mm256_loadu_ps(v+16), d = _mm256_loadu_ps(v+24);
    for (int t = 0; t < batch; ++t, x += stride) {
        __m256 sum = _mm256_add_ps(
            _mm256_add_ps(_mm256_mul_ps(a,_mm256_loadu_ps(x)), _mm256_mul_ps(b,_mm256_loadu_ps(x+8))),
            _mm256_add_ps(_mm256_mul_ps(c,_mm256_loadu_ps(x+16)), _mm256_mul_ps(d,_mm256_loadu_ps(x+24))));
        __m128 h = _mm_add_ps(_mm256_castps256_ps128(sum), _mm256_extractf128_ps(sum,1));
        h = _mm_hadd_ps(h,h);
        out[t] += _mm_cvtss_f32(_mm_hadd_ps(h,h));
    }
}
/* Decode: keep eight partial sums across the entire row. Inlining unpack
 * here also lets the compiler feed SIMD registers directly into the dot. */
__attribute__((target("avx2")))
static float row_avx2(const uint8_t *p, const float *x, Py_ssize_t blocks, int type, int bytes, int width) {
    __m256 y = _mm256_setzero_ps();
    for (; blocks; --blocks, p+=bytes) {
        for (int g=0; g<width/32; ++g, x+=32) {
            float v[32]; unpack_avx2(v,p,type,g);
            for (int j=0; j<32; j+=8)
                y = _mm256_add_ps(y,_mm256_mul_ps(_mm256_loadu_ps(v+j),_mm256_loadu_ps(x+j)));
        }
    }
    __m128 h = _mm_add_ps(_mm256_castps256_ps128(y),_mm256_extractf128_ps(y,1));
    h = _mm_hadd_ps(h,h);
    return _mm_cvtss_f32(_mm_hadd_ps(h,h));
}
#elif defined(HAVE_NEON)
static void unpack_neon(float *v, const uint8_t *p, int type, int g) {
    group_layout a = layout(p,type,g);
    float d = a.d, m = a.m;
    for (int j = 0; j < 32; j += 8) {
        if (j == 16 && type == 14) { d = scale(p+208)*(int8_t)p[193+2*g]; m = -32*d; }
        int off = type < 12 ? j%16 : j;
        uint8x8_t b = vand_u8(vshl_u8(vld1_u8(a.lo+off),vdup_n_s8(type < 12 && j>=16 ? -4 : -a.shift)),
                              vdup_n_u8(15));
        if (a.hi) {
            uint8x8_t h = vand_u8(vshl_u8(vld1_u8(a.hi+j),vdup_n_s8(-a.hshift)),vdup_n_u8(a.hmask));
            b = vorr_u8(b,vshl_n_u8(h,4));
        }
        uint16x8_t u = vmovl_u8(b);
        vst1q_f32(v+j, vmlaq_n_f32(vdupq_n_f32(m),vcvtq_f32_u32(vmovl_u16(vget_low_u16(u))),d));
        vst1q_f32(v+j+4, vmlaq_n_f32(vdupq_n_f32(m),vcvtq_f32_u32(vmovl_u16(vget_high_u16(u))),d));
    }
}
static void group_neon(const float *v, const float *x, Py_ssize_t stride, int batch, float *out) {
    for (int t = 0; t < batch; ++t, x += stride) {
        float32x4_t sum = vdupq_n_f32(0);
        for (int j = 0; j < 32; j += 4) sum = vfmaq_f32(sum,vld1q_f32(v+j),vld1q_f32(x+j));
        out[t] += vaddvq_f32(sum);
    }
}
static float row_neon(const uint8_t *p, const float *x, Py_ssize_t blocks, int type, int bytes, int width) {
    float32x4_t y = vdupq_n_f32(0);
    for (; blocks; --blocks, p+=bytes) {
        for (int g=0; g<width/32; ++g, x+=32) {
            float v[32]; unpack_neon(v,p,type,g);
            for (int j=0; j<32; j+=4) y = vfmaq_f32(y,vld1q_f32(v+j),vld1q_f32(x+j));
        }
    }
    return vaddvq_f32(y);
}
#endif

static PyObject *lowbit_matvec(PyObject *self, PyObject *args) {
    Py_buffer w, x, out;
    int batch, type;
    if (!PyArg_ParseTuple(args, "y*y*w*ii", &w, &x, &out, &batch, &type)) return NULL;
    int bytes = type == 2 ? 18 : type == 3 ? 20 : type == 12 ? 144 :
                type == 13 ? 176 : type == 14 ? 210 : 0;
    int width = type < 12 ? 32 : 256;
    int valid = bytes && batch >= 1 && batch <= 32 && x.len > 0 && out.len > 0
        && !(x.len % (4*width*batch)) && !(out.len % (4*batch))
        && !((uintptr_t)x.buf % _Alignof(float)) && !((uintptr_t)out.buf % _Alignof(float));
    if (valid) {
        Py_ssize_t rows = out.len/(4*batch), cols = x.len/(4*batch);
        Py_ssize_t blocks = cols/width;
        valid = !(w.len % rows) && w.len/rows == blocks*bytes;
        if (valid) {
            Py_BEGIN_ALLOW_THREADS
            const uint8_t *p = w.buf;
            float *y = out.buf;
            for (Py_ssize_t i = 0; i < rows; ++i, y += batch) {
                if (batch == 1 && lowbit_row) {
                    *y = lowbit_row(p,x.buf,blocks,type,bytes,width);
                    p += blocks*bytes;
                    continue;
                }
                memset(y, 0, batch*sizeof(float));
                const float *a = x.buf;
                for (Py_ssize_t k = 0; k < blocks; ++k, p += bytes) {
                    for (int g = 0; g < width/32; ++g, a += 32) {
                        float v[32];
                        unpack(v,p,type,g);
                        accumulate(v,a,cols,batch,y);
                    }
                }
            }
            Py_END_ALLOW_THREADS
        }
    }
    PyBuffer_Release(&out); PyBuffer_Release(&x); PyBuffer_Release(&w);
    if (!valid) { PyErr_SetString(PyExc_ValueError, "invalid low-bit matvec buffers"); return NULL; }
    Py_RETURN_NONE;
}

static PyMethodDef methods[] = {{"matvec", matvec, METH_VARARGS, NULL},
                                {"lowbit_matvec", lowbit_matvec, METH_VARARGS, NULL}, {NULL}};
static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "_q8", NULL, -1, methods};
PyMODINIT_FUNC PyInit__q8(void) {
    for (unsigned h = 0; h < 65536; ++h) {
        unsigned e = (h >> 10) & 31, m = h & 1023;
        float v = e == 31 ? (m ? NAN : INFINITY) : e ? ldexpf(1024 + m, (int)e - 25) : ldexpf(m, -24);
        scales[h] = h & 32768 ? -v : v;
    }
#ifdef HAVE_AVX2
    if (__builtin_cpu_supports("avx2")) {
        dot = avx2; dot4 = avx2_four; accumulate = group_avx2; unpack = unpack_avx2; lowbit_row = row_avx2;
    }
#elif defined(HAVE_NEON)
    dot = neon;
    dot4 = neon_four;
    accumulate = group_neon;
    unpack = unpack_neon;
    lowbit_row = row_neon;
#endif
    return PyModule_Create(&module);
}
