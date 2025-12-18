// mc_pricer.cpp (CPU + OpenMP)
// Monte Carlo pricer for European / Asian / Basket options (CPU baseline).

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

struct OptionParams {
  double S0;
  double K;
  double T;
  double r;
  double sigma;
};

enum class OptionType {
  EuropeanCall = 0,
  AsianArithmeticCall = 1,
  BasketEuropeanCall = 2
};

struct MCConfig {
  OptionType type = OptionType::EuropeanCall;
  int steps = 252;
  int assets = 1;
  double rho = 0.0;
};

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
  auto norm_cdf = [](double x) {
    return 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
  };
  return S * norm_cdf(d1) - K * std::exp(-r * T) * norm_cdf(d2);
}

static std::vector<double> cholesky_equicorrelation(int assets, double rho) {
  std::vector<double> L((size_t)assets * (size_t)assets, 0.0);
  std::vector<double> A((size_t)assets * (size_t)assets, 0.0);
  for (int i = 0; i < assets; ++i) {
    for (int j = 0; j < assets; ++j) {
      A[(size_t)i * assets + j] = (i == j) ? 1.0 : rho;
    }
  }
  for (int i = 0; i < assets; ++i) {
    for (int j = 0; j <= i; ++j) {
      double sum = A[(size_t)i * assets + j];
      for (int k = 0; k < j; ++k) {
        sum -= L[(size_t)i * assets + k] * L[(size_t)j * assets + k];
      }
      if (i == j) {
        if (sum <= 0.0) throw std::runtime_error("Cholesky failed: matrix not SPD");
        L[(size_t)i * assets + j] = std::sqrt(sum);
      } else {
        L[(size_t)i * assets + j] = sum / L[(size_t)j * assets + j];
      }
    }
  }
  return L;
}

static void mc_price_cpu(
    const OptionParams &p,
    const MCConfig &cfg,
    std::uint64_t paths,
    std::uint64_t seed,
    double &out_price,
    double &out_std_error
) {
  if (paths == 0) {
    out_price = 0.0;
    out_std_error = 0.0;
    return;
  }

  int steps = (cfg.steps > 0) ? cfg.steps : 1;
  int assets = (cfg.assets > 0) ? cfg.assets : 1;

  double disc = std::exp(-p.r * p.T);

  std::vector<double> L;
  if (cfg.type == OptionType::BasketEuropeanCall && assets > 1) {
    L = cholesky_equicorrelation(assets, cfg.rho);
  }

  double sum = 0.0;
  double sum2 = 0.0;

#pragma omp parallel
  {
#ifdef _OPENMP
    int tid = omp_get_thread_num();
#else
    int tid = 0;
#endif
    std::mt19937_64 rng(seed + 1315423911ULL * (std::uint64_t)tid);
    std::normal_distribution<double> nd(0.0, 1.0);

    double local_sum = 0.0;
    double local_sum2 = 0.0;

#pragma omp for schedule(static)
    for (long long i = 0; i < (long long)paths; ++i) {
      if (cfg.type == OptionType::EuropeanCall && assets == 1) {
        double z = nd(rng);
        double ST = p.S0 * std::exp((p.r - 0.5 * p.sigma * p.sigma) * p.T + p.sigma * std::sqrt(p.T) * z);
        double payoff = std::max(ST - p.K, 0.0) * disc;
        local_sum += payoff;
        local_sum2 += payoff * payoff;
      } else if (cfg.type == OptionType::AsianArithmeticCall && assets == 1) {
        double dt = p.T / (double)steps;
        double drift = (p.r - 0.5 * p.sigma * p.sigma) * dt;
        double vol = p.sigma * std::sqrt(dt);
        double S = p.S0;
        double sumS = 0.0;
        for (int t = 0; t < steps; ++t) {
          double z = nd(rng);
          S *= std::exp(drift + vol * z);
          sumS += S;
        }
        double avgS = sumS / (double)steps;
        double payoff = std::max(avgS - p.K, 0.0) * disc;
        local_sum += payoff;
        local_sum2 += payoff * payoff;
      } else if (cfg.type == OptionType::BasketEuropeanCall && assets > 1) {
        double dt = p.T / (double)steps;
        double drift = (p.r - 0.5 * p.sigma * p.sigma) * dt;
        double vol = p.sigma * std::sqrt(dt);

        std::vector<double> S(assets, p.S0);
        std::vector<double> z(assets);
        std::vector<double> y(assets);

        for (int t = 0; t < steps; ++t) {
          for (int a = 0; a < assets; ++a) z[a] = nd(rng);
          for (int i = 0; i < assets; ++i) {
            double acc = 0.0;
            int row = i * assets;
            for (int k = 0; k <= i; ++k) acc += L[(size_t)row + k] * z[k];
            y[i] = acc;
          }
          for (int i = 0; i < assets; ++i) {
            S[i] *= std::exp(drift + vol * y[i]);
          }
        }

        double basket = 0.0;
        for (int i = 0; i < assets; ++i) basket += S[i];
        basket /= (double)assets;
        double payoff = std::max(basket - p.K, 0.0) * disc;
        local_sum += payoff;
        local_sum2 += payoff * payoff;
      }
    }

#pragma omp atomic
    sum += local_sum;
#pragma omp atomic
    sum2 += local_sum2;
  }

  double mean_payoff = sum / (double)paths;
  double ex2 = sum2 / (double)paths;
  double var = ex2 - mean_payoff * mean_payoff;
  if (var < 0.0) var = 0.0;

  out_price = mean_payoff;
  out_std_error = std::sqrt(var / (double)paths);
}

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
            << "  --threads N   OpenMP threads (default: OMP default)\n";
}

int main(int argc, char **argv) {
  if (argc == 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
    usage(argv[0]);
    return 0;
  }

  std::string type_str = get_arg(argc, argv, "--type", "european");
  for (auto &c : type_str) c = (char)std::tolower(c);

  OptionType type = OptionType::EuropeanCall;
  if (type_str == "european") type = OptionType::EuropeanCall;
  else if (type_str == "asian") type = OptionType::AsianArithmeticCall;
  else if (type_str == "basket") type = OptionType::BasketEuropeanCall;
  else {
    std::cerr << "Unknown --type: " << type_str << "\n";
    usage(argv[0]);
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
  int threads = std::stoi(get_arg(argc, argv, "--threads", "0"));

#ifdef _OPENMP
  if (threads > 0) {
    omp_set_num_threads(threads);
  }
#endif

  OptionParams p{};
  p.S0 = S0;
  p.K = K;
  p.T = T;
  p.r = r;
  p.sigma = sigma;

  MCConfig cfg;
  cfg.type = type;
  cfg.steps = steps;
  cfg.assets = (type == OptionType::BasketEuropeanCall) ? std::max(2, assets) : 1;
  cfg.rho = rho;

  double price = 0.0, se = 0.0;
  auto t0 = std::chrono::high_resolution_clock::now();
  mc_price_cpu(p, cfg, paths, seed, price, se);
  auto t1 = std::chrono::high_resolution_clock::now();
  double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  std::cout.setf(std::ios::fixed);
  std::cout.precision(6);

  std::string type_name = (type == OptionType::EuropeanCall) ? "EuropeanCall" :
                          (type == OptionType::AsianArithmeticCall) ? "AsianArithmeticCall" :
                          "BasketEuropeanCall";

  std::cout << type_name << " CPU MC (OpenMP)" << "\n";
  std::cout << "S0=" << S0 << " K=" << K << " r=" << r
            << " sigma=" << sigma << " T=" << T;
  if (type != OptionType::EuropeanCall) std::cout << " steps=" << steps;
  if (type == OptionType::BasketEuropeanCall) std::cout << " assets=" << cfg.assets << " rho=" << rho;
  std::cout << " paths=" << paths << "\n";
  std::cout << "price=" << price << "  std_error=" << se
            << "  time_ms=" << ms << "\n";

  if (type == OptionType::EuropeanCall) {
    double bs = black_scholes_call(S0, K, T, r, sigma);
    std::cout << "BS_analytic=" << bs << " (for comparison)" << "\n";
  }

  return 0;
}
