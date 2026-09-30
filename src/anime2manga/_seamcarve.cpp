// Native seam-carving engine for anime2manga, exposed as a CPython extension.
//
// This is the compiled backend behind `anime2manga.seam_carving.carve_width`.
// It is a faithful port of the pure-Python engine in that module: L1 Sobel
// gradient energy, Rubinstein/Shamir/Avidan forward energy, optional
// face-hostile additive saliency, greedy vertical-seam removal, per-seam
// metrics, seam recording and snapshot capture.  Output is bit-identical to the
// Python engine (see tests/test_seam_carving.py).
//
// Loading: rather than a hand-written CPython binding, the module exports the
// address of `carve_width_full`; `seam_carving.py` builds a `ctypes` prototype
// from it. This keeps the engine a plain C ABI that ctypes can call while
// letting setuptools handle compilation, platform tags and installation.
//
// Build: `pip install -e .` (see setup.py); no OpenCV C++ headers are required.

#include <Python.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

namespace {

const float kInf = std::numeric_limits<float>::infinity();

inline int clamp_int(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

// BGR (interleaved uint8) -> gray, matching OpenCV's higher-precision fixed
// point path (B2Y=3735, G2Y=19235, R2Y=9798, shift 15).  The naive 14-bit
// coefficients differ by +-1 on a few pixels.
void make_gray(const uint8_t* bgr, int h, int w, uint8_t* gray) {
    const int n = h * w;
    for (int i = 0; i < n; ++i) {
        const uint8_t* p = bgr + static_cast<size_t>(i) * 3;
        const int y = (p[0] * 3735 + p[1] * 19235 + p[2] * 9798 + 16384) >> 15;
        gray[i] = static_cast<uint8_t>(y);
    }
}

// energy = |dI/dx| + |dI/dy| (3x3 Sobel, BORDER_REFLECT_101), float32.
// BORDER_REFLECT_101 on an axis of length 1 maps every index to 0, so guard the
// h == 1 / w == 1 cases (otherwise the row/column indices fall out of bounds).
void sobel_energy(const uint8_t* gray, int h, int w, float* energy) {
    for (int y = 0; y < h; ++y) {
        const int ym = h == 1 ? 0 : (y == 0 ? 1 : y - 1);
        const int yp = h == 1 ? 0 : (y == h - 1 ? h - 2 : y + 1);
        const uint8_t* r0 = gray + static_cast<size_t>(ym) * w;
        const uint8_t* r1 = gray + static_cast<size_t>(y) * w;
        const uint8_t* r2 = gray + static_cast<size_t>(yp) * w;
        float* e = energy + static_cast<size_t>(y) * w;
        for (int x = 0; x < w; ++x) {
            const int xm = w == 1 ? 0 : (x == 0 ? 1 : x - 1);
            const int xp = w == 1 ? 0 : (x == w - 1 ? w - 2 : x + 1);
            const int gx = -static_cast<int>(r0[xm]) + static_cast<int>(r0[xp]) -
                           2 * static_cast<int>(r1[xm]) + 2 * static_cast<int>(r1[xp]) -
                           static_cast<int>(r2[xm]) + static_cast<int>(r2[xp]);
            const int gy = -static_cast<int>(r0[xm]) - 2 * static_cast<int>(r0[x]) -
                           static_cast<int>(r0[xp]) + static_cast<int>(r2[xm]) +
                           2 * static_cast<int>(r2[x]) + static_cast<int>(r2[xp]);
            e[x] = static_cast<float>(std::abs(gx) + std::abs(gy));
        }
    }
}

// Forward-energy costs (cL, cU, cR) per pixel; edge-replicated borders exactly
// like the numpy `np.pad(mode="edge")` version.
void forward_costs(const uint8_t* gray, int h, int w, float* cL, float* cU, float* cR) {
    for (int y = 0; y < h; ++y) {
        const int ym = y == 0 ? 0 : y - 1;
        const uint8_t* up = gray + static_cast<size_t>(ym) * w;
        const uint8_t* row = gray + static_cast<size_t>(y) * w;
        float* l = cL + static_cast<size_t>(y) * w;
        float* u = cU + static_cast<size_t>(y) * w;
        float* r = cR + static_cast<size_t>(y) * w;
        for (int x = 0; x < w; ++x) {
            const int xm = x == 0 ? 0 : x - 1;
            const int xp = x == w - 1 ? w - 1 : x + 1;
            const float left = static_cast<float>(row[xm]);
            const float right = static_cast<float>(row[xp]);
            const float upv = static_cast<float>(up[x]);
            const float nh = std::abs(left - right);
            l[x] = nh + std::abs(upv - left);
            u[x] = nh;
            r[x] = nh + std::abs(upv - right);
        }
    }
}

// Dynamic program + backtracking; mirrors `find_vertical_seam`.
void find_seam(const float* energy, const float* cL, const float* cU, const float* cR,
               const float* saliency, int h, int w, int* seam, float* added, double* acc,
               int8_t* back) {
    for (int x = 0; x < w; ++x) {
        acc[x] = energy[x] + (saliency ? saliency[x] : 0.0f);
    }
    for (int y = 1; y < h; ++y) {
        const double* prev = acc + static_cast<size_t>(y - 1) * w;
        double* cur = acc + static_cast<size_t>(y) * w;
        int8_t* bk = back + static_cast<size_t>(y) * w;
        const float* ey = energy + static_cast<size_t>(y) * w;
        const float* sl = saliency ? saliency + static_cast<size_t>(y) * w : nullptr;
        const float* fl = cL + static_cast<size_t>(y) * w;
        const float* fu = cU + static_cast<size_t>(y) * w;
        const float* fr = cR + static_cast<size_t>(y) * w;
        for (int x = 0; x < w; ++x) {
            const double lv = x > 0 ? prev[x - 1] : kInf;
            const double uv = prev[x];
            const double rv = x < w - 1 ? prev[x + 1] : kInf;
            const double o0 = lv + fl[x];
            const double o1 = uv + fu[x];
            const double o2 = rv + fr[x];
            // First-minimum tie-break to match np.argmin(left, up, right).
            double best = o0;
            int choice = -1;
            if (o1 < best) {
                best = o1;
                choice = 0;
            }
            if (o2 < best) {
                best = o2;
                choice = 1;
            }
            cur[x] = ey[x] + (sl ? sl[x] : 0.0f) + best;
            bk[x] = static_cast<int8_t>(choice);
        }
    }
    const double* last = acc + static_cast<size_t>(h - 1) * w;
    int best = 0;
    for (int x = 1; x < w; ++x) {
        if (last[x] < last[best]) best = x;
    }
    seam[h - 1] = best;
    for (int y = h - 1; y > 0; --y) {
        seam[y - 1] = seam[y] + back[static_cast<size_t>(y) * w + seam[y]];
    }
    std::memset(added, 0, sizeof(float) * static_cast<size_t>(h));
    for (int y = 1; y < h; ++y) {
        const int x = seam[y];
        const int mv = back[static_cast<size_t>(y) * w + x];
        const size_t idx = static_cast<size_t>(y) * w + x;
        added[y] = mv == -1 ? cL[idx] : (mv == 0 ? cU[idx] : cR[idx]);
    }
}

// Copy `src` (h rows of `w` elements, `elem` bytes each) into `dst` with the
// seam column dropped, packing each row to `w - 1` elements.  `elem` is 3 for
// BGR and 4 for a float32 mask.
void remove_seam_into(const uint8_t* src, uint8_t* dst, int h, int w, int elem,
                      const int* seam) {
    const int out_w = w - 1;
    for (int y = 0; y < h; ++y) {
        const uint8_t* srow = src + static_cast<size_t>(y) * w * elem;
        uint8_t* drow = dst + static_cast<size_t>(y) * out_w * elem;
        const int s = seam[y];
        if (s > 0) std::memcpy(drow, srow, static_cast<size_t>(s) * elem);
        std::memcpy(drow + static_cast<size_t>(s) * elem, srow + static_cast<size_t>(s + 1) * elem,
                    static_cast<size_t>(out_w - s) * elem);
    }
}

// Copy a packed h*w*3 image into a stride-`stride` buffer row by row.
void copy_with_stride(const uint8_t* src, uint8_t* dst, int h, int w, int stride) {
    for (int y = 0; y < h; ++y) {
        std::memcpy(dst + static_cast<size_t>(y) * stride * 3,
                    src + static_cast<size_t>(y) * w * 3, static_cast<size_t>(w) * 3);
    }
}

// Boolean face mask, optionally dilated by OpenCV's MORPH_ELLIPSE kernel.
// Both the radius and the per-row half-width use round-half-to-even
// (std::nearbyint) to match Python's `round()` and OpenCV's `cvRound`.
void build_mask(int h, int w, const int* faces, int n_faces, double dilation, uint8_t* mask) {
    std::memset(mask, 0, static_cast<size_t>(h) * w);
    for (int f = 0; f < n_faces; ++f) {
        int x0 = faces[f * 4 + 0];
        int y0 = faces[f * 4 + 1];
        int x1 = x0 + faces[f * 4 + 2];
        int y1 = y0 + faces[f * 4 + 3];
        x0 = clamp_int(x0, 0, w);
        y0 = clamp_int(y0, 0, h);
        x1 = clamp_int(x1, 0, w);
        y1 = clamp_int(y1, 0, h);
        for (int y = y0; y < y1; ++y) {
            std::memset(mask + static_cast<size_t>(y) * w + x0, 1, x1 - x0);
        }
    }
    if (n_faces == 0 || dilation <= 0.0) return;
    const int radius =
        std::max(1, static_cast<int>(std::nearbyint(dilation * std::min(h, w))));
    std::vector<std::pair<int, int>> offsets;
    for (int dy = -radius; dy <= radius; ++dy) {
        const int dx = static_cast<int>(
            std::nearbyint(std::sqrt(static_cast<double>(radius * radius - dy * dy))));
        for (int d = -dx; d <= dx; ++d) offsets.emplace_back(d, dy);
    }
    std::vector<uint8_t> src(mask, mask + static_cast<size_t>(h) * w);
    for (const auto& off : offsets) {
        for (int y = 0; y < h; ++y) {
            const int yy = y + off.second;
            if (yy < 0 || yy >= h) continue;
            for (int x = 0; x < w; ++x) {
                if (!src[static_cast<size_t>(y) * w + x]) continue;
                const int xx = x + off.first;
                if (xx < 0 || xx >= w) continue;
                mask[static_cast<size_t>(yy) * w + xx] = 1;
            }
        }
    }
}

// Linear-interpolation quantile, matching numpy's float32 result exactly.
// numpy computes pos in float64 but the `_lerp` step in float32 with a
// two-branch formula (`a + (b-a)*t` for t < 0.5, else `b - (b-a)*(1-t)`) to
// limit rounding error; `np.quantile` on a float32 array returns float32.
// Python then widens that float32 to float for the per-pixel comparison.
float quantile(std::vector<float>& values, double q) {
    if (values.empty()) return 0.0f;
    std::sort(values.begin(), values.end());
    const double pos = q * (values.size() - 1);
    const size_t lo = static_cast<size_t>(std::floor(pos));
    const size_t hi = static_cast<size_t>(std::ceil(pos));
    const double frac = pos - lo;
    const float a = values[lo];
    const float b = values[hi];
    const float diff = b - a;
    if (frac < 0.5) {
        return a + diff * static_cast<float>(frac);
    }
    return b - diff * static_cast<float>(1.0 - frac);
}

}  // namespace

extern "C" {

// Per-seam metrics, mirrored by a ctypes.Structure in seam_carving.py.
typedef struct {
    double removed_energy;
    double added_energy;
    long long detail_pixels;
    long long protected_pixels;
    double energy_sum_after;
    int width_after;
    int reserved;
} SeamStepRaw;

// Called after each removal with the carved image (packed h*w*3, valid only for
// the duration of the call) and the step just produced.
typedef void (*seam_callback_t)(int k, const uint8_t* image, int height, int width,
                                double removed_energy, double added_energy, long long detail_pixels,
                                long long protected_pixels, double energy_sum_after,
                                int width_after, void* user);

// Carve `src` down to `target_width`, returning the new width (or -1 on bad
// input).  All outputs are optional except `dst`.
//
// `face_dilation`, `face_energy_factor` and `detail_quantile` are doubles: they
// must not be truncated to float at the ABI, or mask saliency and the detail
// threshold stop matching the Python engine.
//
// `record_seams` writes the per-row column of every removed seam into
// `seams_trace` (n_seams * height ints).
//
// Snapshots: for each `snapshot_ratios[i]` (only ratios > 0, ascending), the
// image is copied into `snapshots_buf` block `i` (each `height * width * 3`
// bytes, row stride `width`) when `k / width0` first reaches it, and
// `snapshot_widths[i]` is set to the width then (or left -1 if never reached).
int carve_width_full(const uint8_t* src, int height, int width, int target_width, const int* faces,
                     int n_faces, int protect_faces, double face_dilation, double face_energy_factor,
                     double detail_quantile, int record_seams, int n_snapshots,
                     const double* snapshot_ratios, uint8_t* snapshots_buf, int* snapshot_widths,
                     uint8_t* dst, int* out_width, int* out_seams, int* seams_trace,
                     SeamStepRaw* steps_out, double* out_base_energy, double* out_energy_sum0,
                     long long* out_detail_total, long long* out_protected_total,
                     seam_callback_t callback, void* user) {
    if (!src || !dst || height < 1 || width < 1) return -1;
    for (int i = 0; i < n_snapshots; ++i) {
        if (snapshot_widths) snapshot_widths[i] = -1;
    }

    const size_t npix = static_cast<size_t>(height) * width;
    std::vector<uint8_t> current(src, src + npix * 3);
    int cur_w = width;
    const int target = clamp_int(target_width, 1, width);
    if (out_width) *out_width = cur_w;
    if (out_seams) *out_seams = 0;

    std::vector<uint8_t> mask;
    float* mask_f = nullptr;
    std::vector<float> mask_f_buf;
    std::vector<float> mask_f_alt;
    if (n_faces > 0) {
        mask.resize(npix);
        build_mask(height, width, faces, n_faces, face_dilation, mask.data());
        mask_f_buf.resize(npix);
        for (size_t i = 0; i < npix; ++i) mask_f_buf[i] = static_cast<float>(mask[i]);
        mask_f = mask_f_buf.data();
    }

    std::vector<uint8_t> gray0(npix);
    std::vector<float> energy0(npix);
    make_gray(src, height, width, gray0.data());
    sobel_energy(gray0.data(), height, width, energy0.data());
    double sum0 = 0.0;
    for (size_t i = 0; i < npix; ++i) sum0 += energy0[i];
    const float base_energy = static_cast<float>(sum0 / npix);
    std::vector<float> copy0(energy0);
    const float detail_threshold = quantile(copy0, detail_quantile);
    const long long detail_total = [&]() {
        long long total = 0;
        for (size_t i = 0; i < npix; ++i) {
            total += static_cast<double>(energy0[i]) >= static_cast<double>(detail_threshold) ? 1 : 0;
        }
        return total;
    }();
    const long long protected_total = [&]() {
        long long total = 0;
        for (uint8_t v : mask) total += v ? 1 : 0;
        return total;
    }();
    if (out_base_energy) *out_base_energy = base_energy;
    if (out_energy_sum0) *out_energy_sum0 = sum0;
    if (out_detail_total) *out_detail_total = detail_total;
    if (out_protected_total) *out_protected_total = protected_total;

    // numpy promotes the Python-float scalar to float32 before multiplying the
    // mask, so compute the product in double and round once to float.
    const float face_energy =
        (protect_faces && mask_f)
            ? static_cast<float>(face_energy_factor * static_cast<double>(base_energy))
            : 0.0f;

    std::vector<uint8_t> other(npix * 3);
    std::vector<uint8_t> gray(npix);
    std::vector<float> energy(npix), cL(npix), cU(npix), cR(npix);
    std::vector<float> saliency;
    std::vector<double> acc(npix);
    std::vector<int8_t> back(npix);
    std::vector<int> seam(height);
    std::vector<float> added(height);
    std::vector<uint8_t> gray_after(npix);
    std::vector<float> scratch(npix);

    int seams = 0;
    int snap_i = 0;
    while (cur_w > target) {
        const size_t cur_npix = static_cast<size_t>(height) * cur_w;
        make_gray(current.data(), height, cur_w, gray.data());
        sobel_energy(gray.data(), height, cur_w, energy.data());
        forward_costs(gray.data(), height, cur_w, cL.data(), cU.data(), cR.data());

        const float* sal = nullptr;
        if (mask_f && face_energy > 0.0f) {
            saliency.resize(cur_npix);
            for (size_t i = 0; i < cur_npix; ++i) saliency[i] = mask_f[i] * face_energy;
            sal = saliency.data();
        }
        find_seam(energy.data(), cL.data(), cU.data(), cR.data(), sal, height, cur_w, seam.data(),
                  added.data(), acc.data(), back.data());
        if (record_seams && seams_trace) {
            std::memcpy(seams_trace + static_cast<size_t>(seams) * height, seam.data(),
                        sizeof(int) * height);
        }

        double removed_energy = 0.0;
        long long detail_pixels = 0;
        long long protected_pixels = 0;
        for (int y = 0; y < height; ++y) {
            const size_t idx = static_cast<size_t>(y) * cur_w + seam[y];
            removed_energy += energy[idx];
            if (static_cast<double>(energy[idx]) >= static_cast<double>(detail_threshold)) {
                ++detail_pixels;
            }
            if (mask_f && mask_f[idx] > 0.0f) ++protected_pixels;
        }
        removed_energy /= height;
        double added_energy = 0.0;
        for (int y = 0; y < height; ++y) added_energy += added[y];
        added_energy /= height;

        remove_seam_into(current.data(), other.data(), height, cur_w, 3, seam.data());
        current.swap(other);
        if (mask_f) {
            mask_f_alt.resize(cur_npix);
            remove_seam_into(reinterpret_cast<const uint8_t*>(mask_f),
                             reinterpret_cast<uint8_t*>(mask_f_alt.data()), height, cur_w,
                             sizeof(float), seam.data());
            mask_f_buf.swap(mask_f_alt);
            mask_f = mask_f_buf.data();
        }
        --cur_w;
        ++seams;

        const size_t after_npix = static_cast<size_t>(height) * cur_w;
        make_gray(current.data(), height, cur_w, gray_after.data());
        sobel_energy(gray_after.data(), height, cur_w, scratch.data());
        double energy_sum_after = 0.0;
        for (size_t i = 0; i < after_npix; ++i) energy_sum_after += scratch[i];

        const double ratio = static_cast<double>(seams) / width;
        SeamStepRaw step;
        step.removed_energy = removed_energy;
        step.added_energy = added_energy;
        step.detail_pixels = detail_pixels;
        step.protected_pixels = protected_pixels;
        step.energy_sum_after = energy_sum_after;
        step.width_after = cur_w;
        step.reserved = 0;
        if (steps_out) steps_out[seams - 1] = step;

        while (snap_i < n_snapshots && ratio >= snapshot_ratios[snap_i]) {
            copy_with_stride(current.data(),
                             snapshots_buf + static_cast<size_t>(snap_i) * height * width * 3,
                             height, cur_w, width);
            if (snapshot_widths) snapshot_widths[snap_i] = cur_w;
            ++snap_i;
        }
        if (callback) {
            callback(seams, current.data(), height, cur_w, removed_energy, added_energy,
                     detail_pixels, protected_pixels, energy_sum_after, cur_w, user);
        }
    }

    copy_with_stride(current.data(), dst, height, cur_w, width);
    if (out_width) *out_width = cur_w;
    if (out_seams) *out_seams = seams;
    return cur_w;
}

}  // extern "C"

// ------------------------------------------------------------------ module ---

static PyObject* carve_addr(PyObject*, PyObject*) {
    return PyLong_FromVoidPtr(reinterpret_cast<void*>(&carve_width_full));
}

static PyMethodDef k_methods[] = {
    {"carve_addr", carve_addr, METH_NOARGS,
     "Return the address of carve_width_full as an int (for ctypes)."},
    {nullptr, nullptr, 0, nullptr},
};

static struct PyModuleDef k_module = {
    PyModuleDef_HEAD_INIT,
    "_seamcarve",
    "Native seam-carving backend (ctypes-callable carve_width_full).",
    -1,
    k_methods,
};

PyMODINIT_FUNC PyInit__seamcarve(void) { return PyModule_Create(&k_module); }
