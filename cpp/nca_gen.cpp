#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <omp.h>
#include <zlib.h>

// NCA dataset generator.
//
// Output format: raw little-endian uint16 memmap, shape (num_seqs, row_len)
// - seq_len = 1024
// - row_len = 1025 (one extra token for next-token shift)
//
// Each row is generated from a unique randomly-sampled rule θ and random init grid.
// We serialize a rollout; each timestep yields (grid/2)^2 patch tokens.
// We truncate the serialized stream to row_len tokens.
//
// Complexity filtering:
// The paper uses gzip compression ratio of serialized sequences.
// Here we support two modes:
// - gzip_on=cells  : gzip on raw cell-state bytes (uint8), similar to paper.
// - gzip_on=tokens : gzip on serialized uint16 patch tokens (useful to match a tokenized corpus).
//
// Filter options:
// - Band-pass: --gzip_min X --gzip_max Y keeps ratios in [X,Y]
// - Percentile: --gzip_percentile P keeps top P% most complex trajectories (ratio >= cutoff)

namespace {

template <typename T>
T clamp(T x, T lo, T hi) {
  return x < lo ? lo : (x > hi ? hi : x);
}

struct Args {
  std::string out_path;
  std::string meta_path;
  int num_seqs = 0;
  int seed = 123;

  int n_colors = 8;   // per-cell alphabet size
  int grid = 12;      // H=W, must be even

  int steps_store = 29;       // timesteps used for stored tokens
  int steps_complexity = 0;   // if >0, timesteps used for complexity filtering (store still uses steps_store)

  // Optional gzip complexity filtering.
  std::string gzip_on = "cells"; // cells | tokens

  // Band-pass filtering.
  double gzip_min = -1.0;
  double gzip_max = -1.0;

  // Percentile filtering.
  double gzip_percentile = 0.0; // e.g. 50 => keep top 50%
  int gzip_level = 9;
  int warmup_rules = 2048;

  double tau = 1e-3; // sampling temperature
};

Args parse_args(int argc, char** argv) {
  Args a;
  auto need = [&](int& i, const char* name) -> std::string {
    if (i + 1 >= argc) throw std::runtime_error(std::string("Missing value for ") + name);
    return std::string(argv[++i]);
  };

  for (int i = 1; i < argc; i++) {
    std::string key = argv[i];
    if (key == "--out") a.out_path = need(i, "--out");
    else if (key == "--meta") a.meta_path = need(i, "--meta");
    else if (key == "--num_seqs") a.num_seqs = std::stoi(need(i, "--num_seqs"));
    else if (key == "--seed") a.seed = std::stoi(need(i, "--seed"));
    else if (key == "--n_colors") a.n_colors = std::stoi(need(i, "--n_colors"));
    else if (key == "--grid") a.grid = std::stoi(need(i, "--grid"));
    else if (key == "--steps") a.steps_store = std::stoi(need(i, "--steps"));
    else if (key == "--steps_store") a.steps_store = std::stoi(need(i, "--steps_store"));
    else if (key == "--steps_complexity") a.steps_complexity = std::stoi(need(i, "--steps_complexity"));

    else if (key == "--gzip_on") a.gzip_on = need(i, "--gzip_on");
    else if (key == "--gzip_min") a.gzip_min = std::stod(need(i, "--gzip_min"));
    else if (key == "--gzip_max") a.gzip_max = std::stod(need(i, "--gzip_max"));

    else if (key == "--gzip_percentile") a.gzip_percentile = std::stod(need(i, "--gzip_percentile"));
    else if (key == "--gzip_level") a.gzip_level = std::stoi(need(i, "--gzip_level"));
    else if (key == "--warmup_rules") a.warmup_rules = std::stoi(need(i, "--warmup_rules"));
    else if (key == "--tau") a.tau = std::stod(need(i, "--tau"));

    else if (key == "--help" || key == "-h") {
      std::cout << "Usage: nca_gen --out OUT.bin --meta OUT.meta.json --num_seqs N [options]\n";
      std::cout << "Options:\n"
                << "  --seed INT\n"
                << "  --n_colors INT (default 8; vocab = n_colors^4)\n"
                << "  --grid INT (default 12; must be even)\n"
                << "  --steps_store INT (default 29; store tokens from this many steps)\n"
                << "  --steps_complexity INT (default 0; if >0, use this many steps for complexity filter)\n"
                << "  --gzip_on cells|tokens (default cells)\n"
                << "  --gzip_min FLOAT --gzip_max FLOAT  (band-pass filter on ratio)\n"
                << "  --gzip_percentile FLOAT (0 disables; keep top P% by ratio)\n"
                << "  --gzip_level INT (1..9, default 9)\n"
                << "  --warmup_rules INT (default 2048)\n"
                << "  --tau FLOAT (default 1e-3)\n";
      std::exit(0);
    } else {
      throw std::runtime_error("Unknown arg: " + key);
    }
  }

  if (a.out_path.empty()) throw std::runtime_error("--out is required");
  if (a.meta_path.empty()) throw std::runtime_error("--meta is required");
  if (a.num_seqs <= 0) throw std::runtime_error("--num_seqs must be > 0");
  if (a.n_colors <= 1) throw std::runtime_error("--n_colors must be > 1");
  if (a.grid <= 0 || (a.grid % 2) != 0) throw std::runtime_error("--grid must be positive and even");
  if (a.steps_store <= 0) throw std::runtime_error("--steps_store must be > 0");
  if (a.steps_complexity < 0) throw std::runtime_error("--steps_complexity must be >= 0");
  if (a.gzip_percentile < 0.0 || a.gzip_percentile > 100.0) throw std::runtime_error("--gzip_percentile must be in [0,100]");
  if (a.gzip_level < 1 || a.gzip_level > 9) throw std::runtime_error("--gzip_level must be in [1,9]");
  if (a.tau <= 0) throw std::runtime_error("--tau must be > 0");

  if (a.gzip_on != "cells" && a.gzip_on != "tokens") throw std::runtime_error("--gzip_on must be 'cells' or 'tokens'");

  bool band = (a.gzip_min >= 0.0 || a.gzip_max >= 0.0);
  if (band) {
    if (a.gzip_min < 0.0 || a.gzip_max < 0.0) throw std::runtime_error("If using gzip band-pass, set both --gzip_min and --gzip_max");
    if (a.gzip_min > a.gzip_max) throw std::runtime_error("--gzip_min must be <= --gzip_max");
  }

  return a;
}

size_t gzip_compress_size(const uint8_t* data, size_t len, int level) {
  z_stream zs;
  std::memset(&zs, 0, sizeof(zs));

  // windowBits=15+16 => gzip wrapper
  int ret = deflateInit2(&zs, level, Z_DEFLATED, 15 + 16, 8, Z_DEFAULT_STRATEGY);
  if (ret != Z_OK) throw std::runtime_error("deflateInit2 failed");

  zs.next_in = const_cast<Bytef*>(reinterpret_cast<const Bytef*>(data));
  zs.avail_in = static_cast<uInt>(len);

  // upper bound for gzip output
  uLong bound = deflateBound(&zs, (uLong)len);
  std::vector<uint8_t> out(bound);
  zs.next_out = out.data();
  zs.avail_out = static_cast<uInt>(out.size());

  ret = deflate(&zs, Z_FINISH);
  if (ret != Z_STREAM_END) {
    deflateEnd(&zs);
    throw std::runtime_error("deflate did not end (buffer too small?)");
  }

  size_t comp_size = zs.total_out;
  deflateEnd(&zs);
  return comp_size;
}

inline double gzip_ratio_percent_bytes(const uint8_t* raw, size_t len, int level) {
  if (len == 0) return 0.0;
  size_t comp = gzip_compress_size(raw, len, level);
  return (double)comp / (double)len * 100.0;
}

inline float relu(float x) { return x > 0.f ? x : 0.f; }

struct Rule {
  int n;
  // conv_w[oc][s][ky][kx], oc=4, s=n, ky=kx=3
  std::vector<float> conv_w; // size 4*n*3*3
  std::array<float, 4> conv_b;

  // mlp: 4 -> 16 -> n
  std::array<std::array<float, 4>, 16> w1;
  std::array<float, 16> b1;
  std::vector<float> w2; // size n*16
  std::vector<float> b2; // size n

  explicit Rule(int n_) : n(n_), conv_w(4 * n_ * 3 * 3), w2(n_ * 16), b2(n_) {}
};

inline size_t idx_conv(int oc, int s, int ky, int kx, int n) {
  return (((size_t)oc * (size_t)n + (size_t)s) * 3 + (size_t)ky) * 3 + (size_t)kx;
}

Rule sample_rule(int n, std::mt19937& rng) {
  Rule r(n);
  std::normal_distribution<float> nd(0.f, 1.f);

  for (auto& w : r.conv_w) w = nd(rng) * 0.5f;
  for (int i = 0; i < 4; i++) r.conv_b[i] = nd(rng) * 0.1f;

  for (int o = 0; o < 16; o++) {
    for (int i = 0; i < 4; i++) r.w1[o][i] = nd(rng) * 0.5f;
    r.b1[o] = nd(rng) * 0.1f;
  }

  for (auto& w : r.w2) w = nd(rng) * 0.5f;
  for (auto& b : r.b2) b = nd(rng) * 0.1f;

  return r;
}

int sample_gumbel_max(const float* logits, int n, double inv_tau, std::mt19937& rng) {
  std::uniform_real_distribution<double> ud(1e-12, 1.0 - 1e-12);
  int best = 0;
  double best_score = -1e300;
  for (int k = 0; k < n; k++) {
    double u = ud(rng);
    double g = -std::log(-std::log(u));
    double score = (double)logits[k] * inv_tau + g;
    if (score > best_score) {
      best_score = score;
      best = k;
    }
  }
  return best;
}

struct Rollout {
  std::vector<uint16_t> tokens;   // serialized patch tokens
  std::vector<uint8_t> raw_cells; // raw cell states for complexity
};

Rollout rollout_once(const Rule& rule, int grid, int steps, std::mt19937& rng, double tau) {
  const int H = grid;
  const int W = grid;
  const int n = rule.n;

  const int patch = 2;
  const int ph = H / patch;
  const int pw = W / patch;
  const int tokens_per_step = ph * pw;

  std::uniform_int_distribution<int> sd(0, n - 1);

  std::vector<uint8_t> cur((size_t)H * (size_t)W);
  std::vector<uint8_t> nxt((size_t)H * (size_t)W);
  for (int i = 0; i < H * W; i++) cur[i] = (uint8_t)sd(rng);

  Rollout out;
  out.tokens.reserve((size_t)steps * (size_t)tokens_per_step);
  out.raw_cells.reserve((size_t)steps * (size_t)H * (size_t)W);

  const double inv_tau = 1.0 / tau;

  auto at = [&](const std::vector<uint8_t>& g, int y, int x) -> uint8_t {
    y = (y % H + H) % H;
    x = (x % W + W) % W;
    return g[(size_t)y * (size_t)W + (size_t)x];
  };

  auto push_tokens_from_grid = [&]() {
    for (int y = 0; y < H; y += 2) {
      for (int x = 0; x < W; x += 2) {
        int a = at(cur, y, x);
        int b = at(cur, y, x + 1);
        int c = at(cur, y + 1, x);
        int d = at(cur, y + 1, x + 1);
        uint16_t tok = (uint16_t)(a + n * b + n * n * c + n * n * n * d);
        out.tokens.push_back(tok);
      }
    }
  };

  std::vector<float> logits((size_t)n);

  for (int t = 0; t < steps; t++) {
    // Raw cells for complexity.
    out.raw_cells.insert(out.raw_cells.end(), cur.begin(), cur.end());

    // Tokens.
    push_tokens_from_grid();

    // Update.
    for (int y = 0; y < H; y++) {
      for (int x = 0; x < W; x++) {
        float conv_out[4] = {rule.conv_b[0], rule.conv_b[1], rule.conv_b[2], rule.conv_b[3]};

        for (int ky = 0; ky < 3; ky++) {
          for (int kx = 0; kx < 3; kx++) {
            int s = (int)at(cur, y + ky - 1, x + kx - 1);
            for (int oc = 0; oc < 4; oc++) {
              conv_out[oc] += rule.conv_w[idx_conv(oc, s, ky, kx, n)];
            }
          }
        }

        float hidden[16];
        for (int h = 0; h < 16; h++) {
          float z = rule.b1[(size_t)h];
          for (int i = 0; i < 4; i++) z += rule.w1[(size_t)h][(size_t)i] * conv_out[i];
          hidden[h] = relu(z);
        }

        for (int k = 0; k < n; k++) {
          float z = rule.b2[(size_t)k];
          const float* wrow = &rule.w2[(size_t)k * 16];
          for (int h = 0; h < 16; h++) z += wrow[h] * hidden[h];
          logits[(size_t)k] = z;
        }

        int ns = sample_gumbel_max(logits.data(), n, inv_tau, rng);
        nxt[(size_t)y * (size_t)W + (size_t)x] = (uint8_t)ns;
      }
    }
    cur.swap(nxt);
  }

  return out;
}

inline double gzip_ratio_cells(const Rollout& ro, int level) {
  return gzip_ratio_percent_bytes(ro.raw_cells.data(), ro.raw_cells.size(), level);
}

inline double gzip_ratio_tokens(const Rollout& ro, int level) {
  const uint8_t* raw = reinterpret_cast<const uint8_t*>(ro.tokens.data());
  size_t len = ro.tokens.size() * sizeof(uint16_t);
  return gzip_ratio_percent_bytes(raw, len, level);
}

} // namespace

int main(int argc, char** argv) {
  try {
    Args args = parse_args(argc, argv);

    const int seq_len = 1024;
    const int row_len = 1025;

    const int patch = 2;
    const int ph = args.grid / patch;
    const int pw = args.grid / patch;
    const int tokens_per_step = ph * pw;

    int steps_store = args.steps_store;
    int steps_complexity = (args.steps_complexity > 0) ? args.steps_complexity : args.steps_store;

    const int total_tokens_store = steps_store * tokens_per_step;
    const int total_tokens_complexity = steps_complexity * tokens_per_step;

    if (total_tokens_store < row_len) {
      throw std::runtime_error("Invalid: steps_store*(grid/2)^2 must be >= 1025");
    }

    int vocab_size = args.n_colors * args.n_colors * args.n_colors * args.n_colors;
    if (vocab_size > 65535) throw std::runtime_error("vocab_size too large for uint16");

    size_t out_elems = (size_t)args.num_seqs * (size_t)row_len;
    size_t out_bytes = out_elems * sizeof(uint16_t);

    int fd = ::open(args.out_path.c_str(), O_CREAT | O_TRUNC | O_RDWR, 0644);
    if (fd < 0) throw std::runtime_error("Failed to open output file");
    if (ftruncate(fd, (off_t)out_bytes) != 0) throw std::runtime_error("ftruncate failed");

    void* mapped = mmap(nullptr, out_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (mapped == MAP_FAILED) throw std::runtime_error("mmap failed");
    auto* out = reinterpret_cast<uint16_t*>(mapped);

    // Complexity cutoff (percentile).
    double gzip_cutoff = -1.0;
    bool use_band = (args.gzip_min >= 0.0 && args.gzip_max >= 0.0);

    auto compute_ratio = [&](const Rollout& ro) -> double {
      if (args.gzip_on == "cells") return gzip_ratio_cells(ro, args.gzip_level);
      return gzip_ratio_tokens(ro, args.gzip_level);
    };

    if (!use_band && args.gzip_percentile > 0.0) {
      std::vector<double> ratios;
      ratios.reserve((size_t)args.warmup_rules);

      std::mt19937 rng(args.seed + 999);
      for (int i = 0; i < args.warmup_rules; i++) {
        Rule rule = sample_rule(args.n_colors, rng);
        Rollout ro = rollout_once(rule, args.grid, steps_complexity, rng, args.tau);
        double r = compute_ratio(ro);
        ratios.push_back(r);
      }
      std::sort(ratios.begin(), ratios.end());

      // keep top p% => cutoff at (1 - p/100) quantile
      double p = args.gzip_percentile;
      int idx = (int)std::floor((1.0 - p / 100.0) * (ratios.size() - 1));
      idx = clamp(idx, 0, (int)ratios.size() - 1);
      gzip_cutoff = ratios[(size_t)idx];
      std::cerr << "[gzip] mode=" << args.gzip_on << " cutoff for top " << p << "%: ratio >= " << gzip_cutoff << "\n";
    }

    if (use_band) {
      std::cerr << "[gzip] mode=" << args.gzip_on << " band-pass: " << args.gzip_min << "%.." << args.gzip_max << "%\n";
    }

    std::atomic<int> counter(0);
    std::atomic<long long> tries(0);

#pragma omp parallel
    {
      int tid = omp_get_thread_num();
      std::mt19937 rng(args.seed + 1337 * (tid + 1));

      while (true) {
        int cur = counter.load(std::memory_order_relaxed);
        if (cur >= args.num_seqs) break;

        tries.fetch_add(1, std::memory_order_relaxed);

        Rule rule = sample_rule(args.n_colors, rng);
        Rollout ro = rollout_once(rule, args.grid, steps_complexity, rng, args.tau);

        // Ensure we have enough tokens to write a row.
        if ((int)ro.tokens.size() < row_len) continue;

        // Apply filter.
        if (use_band) {
          double r = compute_ratio(ro);
          if (!(r >= args.gzip_min && r <= args.gzip_max)) continue;
        } else if (gzip_cutoff >= 0.0) {
          double r = compute_ratio(ro);
          if (r < gzip_cutoff) continue;
        }

        int idx = counter.fetch_add(1, std::memory_order_relaxed);
        if (idx >= args.num_seqs) break;

        uint16_t* row = out + (size_t)idx * (size_t)row_len;
        for (int j = 0; j < row_len; j++) row[j] = ro.tokens[(size_t)j];

        if ((idx + 1) % 256 == 0 && tid == 0) {
          long long t = tries.load(std::memory_order_relaxed);
          std::cerr << "generated " << (idx + 1) << "/" << args.num_seqs << " (tries=" << t << ")\n";
        }
      }
    }

    msync(mapped, out_bytes, MS_SYNC);
    munmap(mapped, out_bytes);
    close(fd);

    long long total_tries = tries.load();
    std::cerr << "Done. num_seqs=" << args.num_seqs << ", tries=" << total_tries
              << ", acceptance=" << (double)args.num_seqs / (double)std::max(1LL, total_tries) << "\n";

    // Write meta.
    {
      std::ofstream mf(args.meta_path);
      if (!mf) throw std::runtime_error("Failed to open meta file");

      long long total_tokens = (long long)args.num_seqs * (long long)seq_len;

      mf << "{\n";
      mf << "  \"dtype\": \"uint16\",\n";
      mf << "  \"endianness\": \"little\",\n";
      mf << "  \"vocab_size\": " << vocab_size << ",\n";
      mf << "  \"seq_len\": " << seq_len << ",\n";
      mf << "  \"row_len\": " << row_len << ",\n";
      mf << "  \"num_seqs\": " << args.num_seqs << ",\n";
      mf << "  \"total_tokens\": " << total_tokens << ",\n";
      mf << "  \"generator\": \"nca\",\n";
      mf << "  \"n_colors\": " << args.n_colors << ",\n";
      mf << "  \"grid_h\": " << args.grid << ",\n";
      mf << "  \"grid_w\": " << args.grid << ",\n";
      mf << "  \"patch\": 2,\n";
      mf << "  \"steps_store\": " << args.steps_store << ",\n";
      mf << "  \"steps_complexity\": " << steps_complexity << ",\n";
      mf << "  \"tau\": " << args.tau << ",\n";
      mf << "  \"gzip_on\": \"" << args.gzip_on << "\",\n";
      mf << "  \"complexity_filter\": \"";
      if (use_band) mf << "band";
      else if (args.gzip_percentile > 0.0) mf << "top" << args.gzip_percentile << "pct";
      else mf << "none";
      mf << "\",\n";
      mf << "  \"gzip_min\": " << args.gzip_min << ",\n";
      mf << "  \"gzip_max\": " << args.gzip_max << ",\n";
      mf << "  \"gzip_level\": " << args.gzip_level << ",\n";
      mf << "  \"seed\": " << args.seed << "\n";
      mf << "}\n";
    }

    return 0;

  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
}
