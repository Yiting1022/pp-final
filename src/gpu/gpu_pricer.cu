// gpu_pricer.cu
// Unified GPU Monte Carlo CLI for European / Asian / Basket options
// using the shared core in mc_pricer.cu, similar to CPU mc_pricer.cpp.

#include <iostream>
#include <cstdint>
#include <string>
#include <chrono>
#include <algorithm>

#include <cuda_runtime.h>

#include "mc_pricer.cu"

static std::string get_arg(int argc, char **argv, const std::string &key, const std::string &def) {
  for (int i = 1; i < argc - 1; ++i) {
    if (std::string(argv[i]) == key) return argv[i + 1];
  }
  return def;
}

static void usage(const char *prog) {
  std::cerr << "Usage: " << prog << " [options]\n"
            << "  --type european|asian|basket (default european)\n"
            << "  --S0 N        spot price (default 100)\n"
            << "  --K N         strike (default 100)\n"
            << "  --r N         risk-free rate (default 0.05)\n"
            << "  --sigma N     volatility (default 0.2)\n"
            << "  --T N         maturity in years (default 1.0)\n"
            << "  --steps N     time steps (asian/basket, default 252)\n"
            << "  --assets N    assets in basket (basket only, default 8)\n"
            << "  --rho N       equicorrelation for basket (default 0.3)\n"
            << "  --paths N     Monte Carlo paths (default 2000000)\n"
            << "  --seed N      RNG seed (default 42)\n"
            << "  --gpus N      GPUs for basket (N>1 => multi-GPU, default 1)\n"
            << "  --block_size N   CUDA threads per block (default 256)\n"
            << "  --blocks_per_sm N CUDA blocks per SM (default 8)\n";
}

struct PartialResult {
  std::uint64_t paths;
  double mean;
  double se;
};

int main(int argc, char **argv) {
  if (argc == 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
    usage(argv[0]);
    return 0;
  }

  std::string type_str = get_arg(argc, argv, "--type", "european");
  for (auto &c : type_str) c = (char)std::tolower(c);

  mc::OptionType type = mc::OptionType::EuropeanCall;
  if (type_str == "european")      type = mc::OptionType::EuropeanCall;
  else if (type_str == "asian")    type = mc::OptionType::AsianArithmeticCall;
  else if (type_str == "basket")   type = mc::OptionType::BasketEuropeanCall;
  else {
    std::cerr << "Unknown --type: " << type_str << " (expected european|asian|basket)" << std::endl;
    return 1;
  }

  double S0 = std::stod(get_arg(argc, argv, "--S0", "100"));
  double K = std::stod(get_arg(argc, argv, "--K", "100"));
  double r = std::stod(get_arg(argc, argv, "--r", "0.05"));
  double sigma = std::stod(get_arg(argc, argv, "--sigma", "0.2"));
  double T = std::stod(get_arg(argc, argv, "--T", "1.0"));
  int steps = std::stoi(get_arg(argc, argv, "--steps", "252"));
  int assets = std::stoi(get_arg(argc, argv, "--assets", "8"));
  double rho = std::stod(get_arg(argc, argv, "--rho", "0.3"));
  std::uint64_t paths = std::stoull(get_arg(argc, argv, "--paths", "2000000"));
  std::uint64_t seed = std::stoull(get_arg(argc, argv, "--seed", "42"));
  int gpus_req = std::stoi(get_arg(argc, argv, "--gpus", "1"));
  int block_size = std::stoi(get_arg(argc, argv, "--block_size", "256"));
  int blocks_per_sm = std::stoi(get_arg(argc, argv, "--blocks_per_sm", "8"));

  if (paths == 0) {
    std::cerr << "paths must be > 0" << std::endl;
    return 1;
  }

  if (type == mc::OptionType::BasketEuropeanCall && assets < 2) {
    assets = 2;
  }

  mc::OptionParams p{};
  p.S0 = S0;
  p.K = K;
  p.T = T;
  p.r = r;
  p.sigma = sigma;

  // Single-GPU European / Asian / Basket
  if (type != mc::OptionType::BasketEuropeanCall || gpus_req <= 1) {
    mc::GpuWorkspace ws;
    mc::init_workspace(ws, block_size, blocks_per_sm);

    mc::MCConfig cfg;
    cfg.type = type;
    cfg.steps = (steps > 0 ? steps : 1);
    cfg.assets = (type == mc::OptionType::BasketEuropeanCall) ? assets : 1;
    cfg.rho = rho;
    cfg.use_jump = 0;

    double price = 0.0, se = 0.0;
    auto t0 = std::chrono::high_resolution_clock::now();

    if (type == mc::OptionType::EuropeanCall) {
      mc::mc_price_gpu(p, paths, seed, ws, price, se);
    } else if (type == mc::OptionType::AsianArithmeticCall) {
      mc::mc_price_gpu_asian_arithmetic(p, cfg, paths, seed, ws, price, se);
    } else { // BasketEuropeanCall, single GPU
      mc::mc_price_gpu_basket_european(p, cfg, paths, seed, ws, price, se);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout.setf(std::ios::fixed);
    std::cout.precision(6);

    std::string type_name = (type == mc::OptionType::EuropeanCall) ? "EuropeanCall" :
                            (type == mc::OptionType::AsianArithmeticCall) ? "AsianArithmeticCall" :
                            "BasketEuropeanCall";

    std::cout << type_name << " GPU MC" << "\n";
    std::cout << "S0=" << S0 << " K=" << K << " r=" << r
              << " sigma=" << sigma << " T=" << T;
    if (type != mc::OptionType::EuropeanCall) std::cout << " steps=" << cfg.steps;
    if (type == mc::OptionType::BasketEuropeanCall) std::cout << " assets=" << cfg.assets << " rho=" << rho;
    std::cout << " paths=" << paths << "\n";
    std::cout << "price=" << price << "  std_error=" << se
              << "  time_ms=" << ms << "\n";

    if (type == mc::OptionType::EuropeanCall) {
      double bs = mc::black_scholes_call(S0, K, T, r, sigma);
      std::cout << "BS_analytic=" << bs << " (for comparison)" << "\n";
    }

    return 0;
  }

  // Multi-GPU basket (type == basket && gpus_req > 1)

  int device_count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&device_count));
  if (device_count <= 0) {
    mc::die("No CUDA devices available for multi-GPU pricer.");
  }

  int num_gpus = (gpus_req > 0) ? std::min(gpus_req, device_count) : device_count;
  if (num_gpus <= 0) num_gpus = 1;

  std::vector<mc::GpuWorkspace> workspaces(num_gpus);
  for (int i = 0; i < num_gpus; ++i) {
    CUDA_CHECK(cudaSetDevice(i));
    mc::init_workspace(workspaces[i], block_size, blocks_per_sm);
  }

  mc::MCConfig cfg;
  cfg.type = mc::OptionType::BasketEuropeanCall;
  cfg.steps = (steps > 0 ? steps : 1);
  cfg.assets = assets;
  cfg.rho = rho;
  cfg.use_jump = 0;

  std::vector<PartialResult> partials;
  partials.reserve(num_gpus);

  auto t0 = std::chrono::high_resolution_clock::now();

  std::uint64_t base = paths / (std::uint64_t)num_gpus;
  std::uint64_t rem = paths % (std::uint64_t)num_gpus;

  for (int i = 0; i < num_gpus; ++i) {
    std::uint64_t paths_i = base + ((std::uint64_t)i < rem ? 1ULL : 0ULL);
    if (paths_i == 0) continue;

    CUDA_CHECK(cudaSetDevice(i));

    double price_i = 0.0, se_i = 0.0;
    std::uint64_t seed_i = seed + (std::uint64_t)i * 1315423911ULL;
    mc::mc_price_gpu_basket_european(p, cfg, paths_i, seed_i, workspaces[i], price_i, se_i);

    partials.push_back(PartialResult{paths_i, price_i, se_i});
  }

  auto t1 = std::chrono::high_resolution_clock::now();
  double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  std::uint64_t total_paths = 0;
  for (const auto &pt : partials) total_paths += pt.paths;

  double mean_num = 0.0;
  for (const auto &pt : partials) mean_num += (double)pt.paths * pt.mean;
  double mean = mean_num / (double)total_paths;

  double ex2_num = 0.0;
  for (const auto &pt : partials) {
    double var_i = pt.se * pt.se * (double)pt.paths;
    double ex2_i = var_i + pt.mean * pt.mean;
    ex2_num += (double)pt.paths * ex2_i;
  }
  double ex2 = ex2_num / (double)total_paths;
  double var = ex2 - mean * mean;
  if (var < 0.0) var = 0.0;
  double se = std::sqrt(var / (double)total_paths);

  std::cout.setf(std::ios::fixed);
  std::cout.precision(6);
  std::cout << "MultiGPU BasketEuropeanCall GPU MC" << "\n";
  std::cout << "devices=" << num_gpus << " total_paths=" << total_paths << "\n";
  std::cout << "S0=" << S0 << " K=" << K << " r=" << r
            << " sigma=" << sigma << " T=" << T
            << " steps=" << cfg.steps << " assets=" << assets
            << " rho=" << rho << "\n";
  std::cout << "price=" << mean << "  std_error=" << se
            << "  time_ms=" << ms << "\n";

  return 0;
}
