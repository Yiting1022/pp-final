// mc_pricer.cu
// Core GPU Monte Carlo pricer (single-asset European call) with a reusable workspace.
// Design goals:
// - Minimal host overhead for many pricing calls (reuse device allocations).
// - Avoid allocating curand states per path (grid-stride loop, states per thread).
// - Provide both price and standard error.

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#ifndef MC_PRICER_CU_INCLUDED
#define MC_PRICER_CU_INCLUDED

namespace mc {

static inline void die(const std::string &msg) {
  std::cerr << "Error: " << msg << "\n";
  std::exit(1);
}

#define CUDA_CHECK(stmt)                                                                           \
  do {                                                                                             \
    cudaError_t err = (stmt);                                                                      \
    if (err != cudaSuccess) {                                                                      \
      std::ostringstream oss;                                                                      \
      oss << #stmt << " failed: " << cudaGetErrorString(err);                                     \
      ::mc::die(oss.str());                                                                        \
    }                                                                                              \
  } while (0)

// atomicAdd(double) is native on sm_60+, provide fallback for older arch.
__device__ inline double atomicAddDouble(double *address, double val) {
#if __CUDA_ARCH__ >= 600
  return atomicAdd(address, val);
#else
  unsigned long long int *address_as_ull = reinterpret_cast<unsigned long long int *>(address);
  unsigned long long int old = *address_as_ull, assumed;
  do {
    assumed = old;
    double sum = __longlong_as_double(assumed) + val;
    old = atomicCAS(address_as_ull, assumed, __double_as_longlong(sum));
  } while (assumed != old);
  return __longlong_as_double(old);
#endif
}

struct OptionParams {
  double S0;
  double K;
  double T;
  double r;
  double sigma;
};

// Option style for Monte Carlo.
enum class OptionType {
  EuropeanCall = 0,
  AsianArithmeticCall = 1,
  BasketEuropeanCall = 2
};

// Extra Monte Carlo controls (time steps, multi-asset basket, correlation).
// For the original single-asset European call, you can simply use the
// defaults: type=EuropeanCall, steps=1, assets=1, rho=0.
struct MCConfig {
  OptionType type = OptionType::EuropeanCall;
  int steps = 1;     // time steps along each path (>=1)
  int assets = 1;    // number of assets (1 => single-asset)
  double rho = 0.0;  // equicorrelation off-diagonal (for basket only)
  // Optional jump-diffusion extension (Merton-style, same params for all assets).
  // If use_jump == 0 or jump_lambda <= 0, we fall back to pure GBM.
  int use_jump = 0;        // 0: GBM (default), 1: jump-diffusion
  double jump_lambda = 0;  // annual jump intensity λ
  double jump_mu = 0;      // mean of log jump size (ln J)
  double jump_sigma = 0;   // std  of log jump size (ln J)
};

// Host-side Black–Scholes (European call) for baseline/exit valuation.
static inline double black_scholes_call(double S, double K, double T, double r, double sigma) {
  if (T <= 0.0) return (S > K) ? (S - K) : 0.0;
  if (sigma <= 0.0) {
    double forward = S * std::exp(r * T);
    double payoff = (forward > K) ? (forward - K) : 0.0;
    return std::exp(-r * T) * payoff;
  }
  double vol_sqrtT = sigma * std::sqrt(T);
  double d1 = (std::log(S / K) + (r + 0.5 * sigma * sigma) * T) / vol_sqrtT;
  double d2 = d1 - vol_sqrtT;

  // N(x) via erf
  auto norm_cdf = [](double x) {
    return 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
  };

  return S * norm_cdf(d1) - K * std::exp(-r * T) * norm_cdf(d2);
}

struct GpuWorkspace {
  curandStatePhilox4_32_10_t *d_states = nullptr; // size = n_threads_total
  double *d_sum = nullptr;                        // sum of payoffs
  double *d_sum2 = nullptr;                       // sum of payoff^2
  uint64_t n_states = 0;

  int block_size = 256;
  int blocks = 0;

  void free_all() {
    if (d_states) CUDA_CHECK(cudaFree(d_states));
    if (d_sum)    CUDA_CHECK(cudaFree(d_sum));
    if (d_sum2)   CUDA_CHECK(cudaFree(d_sum2));
    d_states = nullptr;
    d_sum = nullptr;
    d_sum2 = nullptr;
    n_states = 0;
  }

  ~GpuWorkspace() { free_all(); }
};

__global__ void setup_rng(curandStatePhilox4_32_10_t *states, uint64_t seed, uint64_t n_states) {
  uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n_states) return;
  // sequence = tid gives independent substreams
  curand_init((unsigned long long)seed, (unsigned long long)tid, 0ULL, &states[tid]);
}

static constexpr int MAX_ASSETS = 32;

// General Monte Carlo kernel supporting:
// - single-asset European call (type=EuropeanCall, assets=1)
// - single-asset Asian arithmetic call (type=AsianArithmeticCall, assets=1)
// - basket European call with equicorrelation (type=BasketEuropeanCall, assets>=1)
// Time discretization is controlled by "steps".
__global__ void mc_kernel_general(
    OptionParams p,
    int opt_type,              // 0: European, 1: Asian, 2: Basket
    int steps,
    int assets,
  int use_jump,
  double jump_lambda,
  double jump_mu,
  double jump_sigma,
    const double * __restrict__ chol_L, // assets x assets lower-triangular (row-major); may be null if assets==1
    curandStatePhilox4_32_10_t *states,
    uint64_t n_states,
    uint64_t n_paths,
    double *sum_payoff,
    double *sum_payoff2
) {
  uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n_states) return;

  curandStatePhilox4_32_10_t local = states[tid];

  if (steps <= 0) steps = 1;
  if (assets <= 0) assets = 1;

  const double dt = p.T / (double)steps;
  const double drift = (p.r - 0.5 * p.sigma * p.sigma) * dt;
  const double vol   = p.sigma * sqrt(dt);
  const double disc  = exp(-p.r * p.T);
  const double lam_dt = (use_jump && jump_lambda > 0.0) ? (jump_lambda * dt) : 0.0;

  double local_sum = 0.0;
  double local_sum2 = 0.0;

  // grid-stride over paths
  for (uint64_t path = tid; path < n_paths; path += n_states) {
    if (assets == 1) {
      // Single-asset case (European or Asian).
      double S = p.S0;
      double avgS = 0.0;

      int j = 0;
      for (; j + 4 <= steps; j += 4) {
        float4 z4 = curand_normal4(&local);
        S *= exp(drift + vol * (double)z4.x);
        if (lam_dt > 0.0) {
          double u = curand_uniform_double(&local);
          if (u < lam_dt) {
            double zJ = curand_normal_double(&local);
            double J = exp(jump_mu + jump_sigma * zJ);
            S *= J;
          }
        }
        if (opt_type == 1) avgS += S;
        S *= exp(drift + vol * (double)z4.y);
        if (lam_dt > 0.0) {
          double u = curand_uniform_double(&local);
          if (u < lam_dt) {
            double zJ = curand_normal_double(&local);
            double J = exp(jump_mu + jump_sigma * zJ);
            S *= J;
          }
        }
        if (opt_type == 1) avgS += S;
        S *= exp(drift + vol * (double)z4.z);
        if (lam_dt > 0.0) {
          double u = curand_uniform_double(&local);
          if (u < lam_dt) {
            double zJ = curand_normal_double(&local);
            double J = exp(jump_mu + jump_sigma * zJ);
            S *= J;
          }
        }
        if (opt_type == 1) avgS += S;
        S *= exp(drift + vol * (double)z4.w);
        if (lam_dt > 0.0) {
          double u = curand_uniform_double(&local);
          if (u < lam_dt) {
            double zJ = curand_normal_double(&local);
            double J = exp(jump_mu + jump_sigma * zJ);
            S *= J;
          }
        }
        if (opt_type == 1) avgS += S;
      }
      for (; j < steps; ++j) {
        float z = curand_normal(&local);
        S *= exp(drift + vol * (double)z);
        if (lam_dt > 0.0) {
          double u = curand_uniform_double(&local);
          if (u < lam_dt) {
            double zJ = curand_normal_double(&local);
            double J = exp(jump_mu + jump_sigma * zJ);
            S *= J;
          }
        }
        if (opt_type == 1) avgS += S;
      }

      double payoff = 0.0;
      if (opt_type == 0 || opt_type == 2) {
        payoff = fmax(S - p.K, 0.0);
      } else {
        double meanS = avgS / (double)steps;
        payoff = fmax(meanS - p.K, 0.0);
      }
      payoff *= disc;

      local_sum  += payoff;
      local_sum2 += payoff * payoff;
    } else {
      // Multi-asset basket option on arithmetic mean of terminal prices.
      double Svec[MAX_ASSETS];
      for (int a = 0; a < assets; ++a) Svec[a] = p.S0;

      double zvec[MAX_ASSETS];
      double yvec[MAX_ASSETS];

      for (int t = 0; t < steps; ++t) {
        int a = 0;
        for (; a + 4 <= assets; a += 4) {
          float4 z4 = curand_normal4(&local);
          zvec[a + 0] = (double)z4.x;
          zvec[a + 1] = (double)z4.y;
          zvec[a + 2] = (double)z4.z;
          zvec[a + 3] = (double)z4.w;
        }
        for (; a < assets; ++a) {
          zvec[a] = (double)curand_normal(&local);
        }

        // Correlate: y = L * z, L lower-triangular.
        for (int i = 0; i < assets; ++i) {
          double acc = 0.0;
          int row = i * assets;
          for (int k = 0; k <= i; ++k) acc += chol_L[row + k] * zvec[k];
          yvec[i] = acc;
        }

        for (int i = 0; i < assets; ++i) {
          Svec[i] *= exp(drift + vol * yvec[i]);
          if (lam_dt > 0.0) {
            double u = curand_uniform_double(&local);
            if (u < lam_dt) {
              double zJ = curand_normal_double(&local);
              double J = exp(jump_mu + jump_sigma * zJ);
              Svec[i] *= J;
            }
          }
        }
      }

      double basket = 0.0;
      for (int i = 0; i < assets; ++i) basket += Svec[i];
      basket /= (double)assets;

      double payoff = fmax(basket - p.K, 0.0) * disc;
      local_sum  += payoff;
      local_sum2 += payoff * payoff;
    }
  }

  states[tid] = local;

  atomicAddDouble(sum_payoff,  local_sum);
  atomicAddDouble(sum_payoff2, local_sum2);
}

// Initialize reusable workspace.
// blocks is typically: SM_count * blocks_per_sm
static inline void init_workspace(GpuWorkspace &ws, int block_size, int blocks_per_sm) {
  ws.free_all();

  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  int sm = prop.multiProcessorCount;

  ws.block_size = (block_size > 0) ? block_size : 256;
  int blocks = (blocks_per_sm > 0) ? (sm * blocks_per_sm) : (sm * 8);
  ws.blocks = blocks;

  ws.n_states = (uint64_t)ws.block_size * (uint64_t)ws.blocks;

  CUDA_CHECK(cudaMalloc(&ws.d_states, ws.n_states * sizeof(curandStatePhilox4_32_10_t)));
  CUDA_CHECK(cudaMalloc(&ws.d_sum, sizeof(double)));
  CUDA_CHECK(cudaMalloc(&ws.d_sum2, sizeof(double)));
}

// Internal helper to run the general kernel with a given configuration.
static inline void mc_price_gpu_general(
    const OptionParams &p,
    const MCConfig &cfg,
    uint64_t paths,
    uint64_t seed,
    GpuWorkspace &ws,
    double &out_price,
    double &out_std_error
) {
  if (!ws.d_states || !ws.d_sum || !ws.d_sum2 || ws.n_states == 0) {
    die("GpuWorkspace not initialized. Call init_workspace(ws, block_size, blocks_per_sm) first.");
  }
  if (paths == 0) {
    out_price = 0.0;
    out_std_error = 0.0;
    return;
  }

  CUDA_CHECK(cudaMemset(ws.d_sum, 0, sizeof(double)));
  CUDA_CHECK(cudaMemset(ws.d_sum2, 0, sizeof(double)));

  setup_rng<<<ws.blocks, ws.block_size>>>(ws.d_states, seed, ws.n_states);
  CUDA_CHECK(cudaGetLastError());

  int steps = (cfg.steps > 0) ? cfg.steps : 1;
  int assets = (cfg.assets > 0) ? cfg.assets : 1;
  if (assets > MAX_ASSETS) {
    die("assets too large for mc_pricer-2 (max 32)");
  }

  int use_jump = (cfg.use_jump != 0 && cfg.jump_lambda > 0.0) ? 1 : 0;
  double jump_lambda = (use_jump ? cfg.jump_lambda : 0.0);
  double jump_mu = cfg.jump_mu;
  double jump_sigma = cfg.jump_sigma;

  // For single-asset cases, we do not need any correlation matrix.
  double *d_L = nullptr;

  // Multi-asset basket requires a Cholesky of the equicorrelation matrix.
  if (assets > 1) {
    // CPU-side construction of equicorrelation matrix and Cholesky factor.
    std::vector<double> A((size_t)assets * (size_t)assets, 0.0);
    for (int i = 0; i < assets; ++i) {
      for (int j = 0; j < assets; ++j) {
        A[(size_t)i * assets + j] = (i == j) ? 1.0 : cfg.rho;
      }
    }

    auto cholesky_lower = [&](const std::vector<double> &M) {
      std::vector<double> L((size_t)assets * (size_t)assets, 0.0);
      for (int i = 0; i < assets; ++i) {
        for (int j = 0; j <= i; ++j) {
          double sum = M[(size_t)i * assets + j];
          for (int k = 0; k < j; ++k) {
            sum -= L[(size_t)i * assets + k] * L[(size_t)j * assets + k];
          }
          if (i == j) {
            if (sum <= 0.0) die("Cholesky failed: matrix not SPD (try different rho/assets).");
            L[(size_t)i * assets + j] = std::sqrt(sum);
          } else {
            L[(size_t)i * assets + j] = sum / L[(size_t)j * assets + j];
          }
        }
      }
      return L;
    };

    std::vector<double> L = cholesky_lower(A);
    CUDA_CHECK(cudaMalloc(&d_L, sizeof(double) * (size_t)assets * (size_t)assets));
    CUDA_CHECK(cudaMemcpy(d_L, L.data(), sizeof(double) * (size_t)assets * (size_t)assets, cudaMemcpyHostToDevice));
  }

  int opt_type = 0;
  switch (cfg.type) {
    case OptionType::EuropeanCall:        opt_type = 0; break;
    case OptionType::AsianArithmeticCall: opt_type = 1; break;
    case OptionType::BasketEuropeanCall:  opt_type = 2; break;
  }

  mc_kernel_general<<<ws.blocks, ws.block_size>>>(
      p,
      opt_type,
      steps,
      assets,
      use_jump,
      jump_lambda,
      jump_mu,
      jump_sigma,
      d_L,
      ws.d_states,
      ws.n_states,
      paths,
      ws.d_sum,
      ws.d_sum2
  );
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  if (d_L) CUDA_CHECK(cudaFree(d_L));

  double sum = 0.0, sum2 = 0.0;
  CUDA_CHECK(cudaMemcpy(&sum, ws.d_sum, sizeof(double), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&sum2, ws.d_sum2, sizeof(double), cudaMemcpyDeviceToHost));

  // Kernel already computes discounted payoff; just take the mean.
  const double mean_payoff = sum / (double)paths;
  out_price = mean_payoff;

  // Standard error of discounted payoff:
  // SE(price) = sqrt( Var(payoff) / paths ), Var = E[x^2] - E[x]^2
  double ex2 = sum2 / (double)paths;
  double var = ex2 - mean_payoff * mean_payoff;
  if (var < 0.0) var = 0.0; // numeric guard
  out_std_error = std::sqrt(var / (double)paths);
}

// Backwards-compatible wrapper: original single-asset European call using
// closed-form terminal distribution (equivalent to steps=1, assets=1).
static inline void mc_price_gpu(
    const OptionParams &p,
    uint64_t paths,
    uint64_t seed,
    GpuWorkspace &ws,
    double &out_price,
    double &out_std_error
) {
  MCConfig cfg;
  cfg.type = OptionType::EuropeanCall;
  cfg.steps = 1;
  cfg.assets = 1;
  cfg.rho = 0.0;
  mc_price_gpu_general(p, cfg, paths, seed, ws, out_price, out_std_error);
}

// Convenience wrappers for additional algorithms.
static inline void mc_price_gpu_european(
    const OptionParams &p,
    const MCConfig &cfg,
    uint64_t paths,
    uint64_t seed,
    GpuWorkspace &ws,
    double &out_price,
    double &out_std_error
) {
  MCConfig c = cfg;
  c.type = OptionType::EuropeanCall;
  mc_price_gpu_general(p, c, paths, seed, ws, out_price, out_std_error);
}

static inline void mc_price_gpu_asian_arithmetic(
    const OptionParams &p,
    const MCConfig &cfg,
    uint64_t paths,
    uint64_t seed,
    GpuWorkspace &ws,
    double &out_price,
    double &out_std_error
) {
  MCConfig c = cfg;
  c.type = OptionType::AsianArithmeticCall;
  c.assets = 1; // only single-asset currently
  mc_price_gpu_general(p, c, paths, seed, ws, out_price, out_std_error);
}

static inline void mc_price_gpu_basket_european(
    const OptionParams &p,
    const MCConfig &cfg,
    uint64_t paths,
    uint64_t seed,
    GpuWorkspace &ws,
    double &out_price,
    double &out_std_error
) {
  MCConfig c = cfg;
  c.type = OptionType::BasketEuropeanCall;
  if (c.assets <= 1) c.assets = 2; // basket needs >=2 assets for non-trivial case
  mc_price_gpu_general(p, c, paths, seed, ws, out_price, out_std_error);
}

} // namespace mc

#endif // MC_PRICER_CU_INCLUDED
