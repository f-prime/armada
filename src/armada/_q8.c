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
#endif

static PyObject *matvec(PyObject *self, PyObject *args) {
    Py_buffer w, x, out;
    if (!PyArg_ParseTuple(args, "y*y*w*", &w, &x, &out)) return NULL;
    Py_ssize_t blocks = x.len / (32 * sizeof(float)), rows = out.len / sizeof(float);
    int valid = blocks > 0 && rows > 0 && x.len % 128 == 0 && out.len % 4 == 0
        && w.len % rows == 0 && w.len / rows == blocks * 34
        && (uintptr_t)x.buf % _Alignof(float) == 0 && (uintptr_t)out.buf % _Alignof(float) == 0;
    if (valid) {
        Py_BEGIN_ALLOW_THREADS
        for (Py_ssize_t i = 0; i < rows; ++i)
            ((float *)out.buf)[i] = dot((const uint8_t *)w.buf + i*blocks*34, x.buf, blocks);
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

static PyMethodDef methods[] = {{"matvec", matvec, METH_VARARGS, NULL}, {NULL}};
static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "_q8", NULL, -1, methods};
PyMODINIT_FUNC PyInit__q8(void) {
    for (unsigned h = 0; h < 65536; ++h) {
        unsigned e = (h >> 10) & 31, m = h & 1023;
        float v = e == 31 ? (m ? NAN : INFINITY) : e ? ldexpf(1024 + m, (int)e - 25) : ldexpf(m, -24);
        scales[h] = h & 32768 ? -v : v;
    }
#ifdef HAVE_AVX2
    if (__builtin_cpu_supports("avx2")) dot = avx2;
#elif defined(HAVE_NEON)
    dot = neon;
#endif
    return PyModule_Create(&module);
}
