// Reproducible fake-data-injection experiments. No third-party dependencies.
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

constexpr int K = 10;
constexpr double PI = 3.14159265358979323846;
using Counts = std::array<long long, K>;
using Values = std::array<double, K>;

void require(bool ok, const std::string& why) {
    if (!ok) throw std::runtime_error(why);
}

struct Random {
    uint64_t state;
    bool cached = false;
    double spare = 0;
    explicit Random(uint64_t seed) : state(seed) {}
    uint64_t next() {
        uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }
    double uniform() { return double(next() >> 11) * 0x1.0p-53; }
    double normal() {
        if (cached) { cached = false; return spare; }
        double radius = std::sqrt(-2 * std::log(1 - uniform()));
        double angle = 2 * PI * uniform();
        spare = radius * std::sin(angle);
        cached = true;
        return radius * std::cos(angle);
    }
};

struct Config {
    std::string learner, grid;
    long long horizon = 100000;
    double delta0 = 0.2, delta = 0.05, lower = -64, upper = 1;
    int batch = 10, trial = 0;
    uint64_t seed = 0;
    double x = 0;
    bool constant_rewards = false;
    double sigma() const { return learner == "ucb" ? 0.5 : 0.05; }
};

struct Result {
    long long time = 0, target = 0, injected = 0;
    double cost = 0;
    int infeasible = 0, planned = 0, recertifications = 0;
    Counts fake{}, genuine{}, prescribed{};
    std::vector<std::pair<long long, double>> trace;
};

struct State {
    Config c;
    Values truth, sums{}, inverse_roots{};
    Counts counts{};
    Result r;
    std::vector<Random> rewards;
    Random posterior, schedule;
    explicit State(Config cfg, Values mu) : c(cfg), truth(mu),
        posterior(c.seed ^ 0xaadc870ef831d633ULL),
        schedule(c.seed ^ 0x32c6a2e0fa61e011ULL) {
        for (int i = 0; i < K; ++i) rewards.emplace_back(c.seed + (i + 1) * 0x123456789abcdefULL);
    }
    double mean(int i) const { return counts[i] ? sums[i] / counts[i] : 0; }
    double beta(long long n) const {
        require(n > 0, "confidence bound before initialization");
        return c.sigma() * std::sqrt(2.0 / n * std::log(PI * PI * K * double(n) * n / (3 * c.delta)));
    }
    void record(int arm, double reward, bool fake, bool bounded = true) {
        require(r.time < c.horizon, "horizon overrun");
        require(arm >= 0 && arm < K && std::isfinite(reward), "invalid record");
        if (fake) {
            require(arm != K - 1, "target-arm injection");
            require(!bounded || (reward >= c.lower && reward <= c.upper), "out-of-range fake reward");
            ++r.injected; ++r.fake[arm];
            r.cost += std::abs(reward - truth[arm]);
        } else { ++r.genuine[arm]; if (arm == K - 1) ++r.target; }
        ++counts[arm]; sums[arm] += reward;
        inverse_roots[arm] = 1.0 / std::sqrt(double(counts[arm]));
        ++r.time;
        if (c.grid == "ratio" && (r.time <= 100 || r.time % 100 == 0 || r.time == c.horizon))
            r.trace.emplace_back(r.time, double(r.target) / r.time);
    }
    void genuine(int i) {
        double reward = truth[i];
        if (!c.constant_rewards) reward = c.learner == "ucb"
            ? double(rewards[i].uniform() < truth[i])
            : truth[i] + c.sigma() * rewards[i].normal();
        record(i, reward, false);
    }
    void initialize() { for (int i = 0; i < K && r.time < c.horizon; ++i) genuine(i); }
    int choose() {
        int best = 0;
        double score = -std::numeric_limits<double>::infinity();
        double scale = 3 * c.sigma() * std::sqrt(std::log(double(r.time + 1)));
        for (int i = 0; i < K; ++i) {
            double value = mean(i) + inverse_roots[i] *
                (c.learner == "ucb" ? scale : posterior.normal());
            if (value > score) { score = value; best = i; }
        }
        return best;
    }
    double benchmark(int i) const {
        double gap = 3 * c.sigma() * c.delta0;
        if (c.learner == "ts") {
            double gamma = std::sqrt(2 * std::log(PI * PI * K / (3 * c.delta))
                                      + 4 * counts[i] * c.delta0 * c.delta0);
            gap = gamma * (inverse_roots[i] + inverse_roots[K - 1]);
        }
        return mean(K - 1) - 2 * beta(counts[K - 1]) - gap;
    }
    long long budget(int i, double ell) const {
        if (mean(i) <= ell) return 0;
        if (ell <= c.lower) return -1;
        double value = (mean(i) - ell) * counts[i] / (ell - c.lower);
        require(std::isfinite(value) && value < 1e12, "unrepresentable injection budget");
        return static_cast<long long>(std::ceil(value));
    }
    // This is a certificate, not a repair of the unresolved PBI theorem.
    // A signed gap and a positive integer cooldown are required at every stage.
    bool cooldown(int i, long long remaining, long long& wait) const {
        if (remaining <= 0) { wait = 0; return true; }
        double lower_target = mean(K - 1) - 2 * beta(counts[K - 1]);
        double admissible = double(c.horizon);
        long long periods = (remaining + c.batch - 1) / c.batch;
        for (long long period = 1; period <= periods; ++period) {
            long long added = std::min(remaining, period * c.batch);
            double n = double(counts[i] + added);
            double post = (sums[i] + added * c.lower) / n;
            double gap = lower_target - post;
            if (!(gap > 0)) return false;
            double log_deadline;
            if (c.learner == "ucb") log_deadline = gap * gap * n / (9 * c.sigma() * c.sigma());
            else {
                double radius = 1 / std::sqrt(n) + inverse_roots[K - 1];
                log_deadline = gap * gap / (4 * radius * radius)
                    - 0.5 * std::log(PI * PI * K / (3 * c.delta));
            }
            double cap = double(r.time) + added + double(period) * c.horizon;
            double candidate = double(c.horizon);
            if (log_deadline < std::log(cap))
                candidate = (std::exp(log_deadline) - r.time - added) / period;
            admissible = std::min(admissible, candidate);
            if (admissible < 1) return false;
        }
        wait = static_cast<long long>(std::floor(std::nextafter(admissible, -std::numeric_limits<double>::infinity())));
        return wait >= 1;
    }
    Result finish() {
        require(r.time == c.horizon, "incomplete run");
        require(std::accumulate(counts.begin(), counts.end(), 0LL) == r.time, "record-count mismatch");
        require(std::accumulate(r.fake.begin(), r.fake.end(), 0LL) == r.injected, "fake-count mismatch");
        require(std::accumulate(r.genuine.begin(), r.genuine.end(), 0LL) + r.injected == r.time,
                "genuine/fake mass balance");
        require(r.fake[K - 1] == 0 && r.target == r.genuine[K - 1], "target accounting");
        return r;
    }
};

Result simulate(const Config& c, const Values& means, const std::string& method,
                const Counts& matched_budget = Counts{}) {
    State s(c, means); s.initialize();
    if (s.r.time == c.horizon) return s.finish();
    long long threshold = std::max(1LL, static_cast<long long>(std::ceil(std::log(double(c.horizon)) / (c.delta0 * c.delta0))));
    Counts remaining{}, due{}, waits{};
    std::array<bool, K> planned{};
    if (method == "heuristic") remaining = matched_budget;
    while (s.r.time < c.horizon) {
        if (method != "heuristic") {
            for (int i = 0; i < K - 1; ++i) if (!planned[i] && s.counts[i] >= threshold) {
                planned[i] = true; ++s.r.planned;
                double ell = s.benchmark(i);
                if (method == "li") {
                    s.r.prescribed[i] = 1;
                    s.record(i, (s.counts[i] + 1) * ell - s.sums[i], true, false);
                } else {
                    long long n = s.budget(i, ell);
                    if (n < 0) { ++s.r.infeasible; continue; }
                    s.r.prescribed[i] = n;
                    if (method == "sbi") {
                        while (n-- > 0 && s.r.time < c.horizon) s.record(i, c.lower, true);
                    } else if (n > 0) {
                        if (!s.cooldown(i, n, waits[i])) { ++s.r.infeasible; continue; }
                        remaining[i] = n; due[i] = s.r.time;
                    }
                }
                if (s.r.time == c.horizon) break;
            }
        }
        if (s.r.time == c.horizon) break;
        if (method == "pbi") {
            int selected = -1;
            for (int i = 0; i < K - 1; ++i)
                if (remaining[i] > 0 && due[i] <= s.r.time &&
                    (selected < 0 || due[i] < due[selected])) selected = i;
            if (selected >= 0) {
                int i = selected;
                if (s.r.time > due[i]) {
                    ++s.r.recertifications;
                    if (!s.cooldown(i, remaining[i], waits[i])) {
                        ++s.r.infeasible; remaining[i] = 0; continue;
                    }
                }
                long long take = std::min<long long>(remaining[i], c.batch);
                for (long long j = 0; j < take && s.r.time < c.horizon; ++j) {
                    s.record(i, c.lower, true); --remaining[i];
                }
                due[i] = s.r.time + waits[i];
                continue;
            }
        } else if (method == "heuristic") {
            long long total = std::accumulate(remaining.begin(), remaining.end(), 0LL);
            long long slots = c.horizon - s.r.time;
            require(total <= slots, "unspendable matched budget");
            if (total > 0 && s.schedule.uniform() < double(total) / slots) {
                long long pick = static_cast<long long>(s.schedule.uniform() * total);
                int arm = 0;
                while (pick >= remaining[arm]) { pick -= remaining[arm]; ++arm; }
                --remaining[arm]; s.record(arm, c.lower, true); continue;
            }
        }
        s.genuine(s.choose());
    }
    Result out = s.finish();
    if (method == "heuristic") {
        require(out.fake == matched_budget, "heuristic must match realized SBI counts per arm");
    }
    return out;
}

Values instance(const Config& c, const std::vector<double>& movie_means) {
    Values means;
    if (c.learner == "ts") for (int i = 0; i < K; ++i) means[i] = 0.9 - 0.05 * i;
    else {
        Random rng(c.seed ^ 0x4fed338f812f0011ULL);
        std::vector<size_t> indices(movie_means.size());
        std::iota(indices.begin(), indices.end(), 0);
        for (int i = 0; i < K; ++i) {
            size_t j = i + size_t(rng.uniform() * (indices.size() - i));
            std::swap(indices[i], indices[j]); means[i] = movie_means[indices[i]];
        }
        std::sort(means.begin(), means.end(), std::greater<double>());
    }
    return means;
}

void self_test() {
    Config c; c.learner = "ucb"; c.horizon = 10000; c.seed = 73; c.constant_rewards = true;
    Values means; for (int i = 0; i < K; ++i) means[i] = 0.9 - 0.05 * i;
    Result sbi = simulate(c, means, "sbi");
    Result random = simulate(c, means, "heuristic", sbi.fake);
    require(sbi.fake == random.fake && std::abs(sbi.cost - random.cost) < 1e-7, "matched-cost control");
    long long threshold = static_cast<long long>(std::ceil(std::log(double(c.horizon)) / (c.delta0 * c.delta0)));
    for (int i = 0; i < K - 1; ++i) require(sbi.genuine[i] <= threshold, "SI deterministic suppression");
    State ledger(c, means); ledger.record(0, 1, false); ledger.record(0, -2, true);
    require(ledger.r.time == 2 && ledger.counts[0] == 2 && ledger.r.injected == 1 &&
            std::abs(ledger.r.cost - 2.9) < 1e-12, "true-mean cost and clock");
    bool rejected = false;
    try { ledger.record(0, c.lower - 1, true); } catch (const std::exception&) { rejected = true; }
    require(rejected, "bounded validation");
    Config counter = c; counter.horizon = 1000000; counter.lower = -1000;
    counter.delta = 0.1; counter.batch = 1;
    State invalid(counter, means); invalid.counts.fill(1); invalid.inverse_roots.fill(1);
    invalid.counts[0] = 346; invalid.sums[0] = 3460; invalid.r.time = 355;
    invalid.inverse_roots[0] = 1 / std::sqrt(346.0);
    long long wait = 0;
    require(!invalid.cooldown(0, 6, wait), "negative-gap PBI counterexample must be rejected");
    for (std::string learner : {"ucb", "ts"}) for (long long horizon : {1, 10, 11, 23, 2000}) {
        Config edge = c; edge.learner = learner; edge.horizon = horizon;
        Result matched = simulate(edge, means, "sbi");
        for (std::string method : {"li", "pbi", "heuristic"}) simulate(edge, means, method, matched.fake);
    }
    std::cout << "PASS: true-mean costs, shared clock, bounded rewards, SI suppression, matched budgets, "
                 "PBI sign checks, initialization, and horizon truncation\n";
}

int main(int argc, char** argv) {
    try {
        std::string means_path = "experiments/data/movielens-means.csv", output = "results/horizon";
        int threads = 6, default_batch = 10; bool quick = false, benchmark = false, ratio_only = false, sweeps = false;
        double fake_lower = -64;
        uint64_t seed_base = 3141592653ULL;
        for (int a = 1; a < argc; ++a) {
            std::string arg = argv[a];
            if (arg == "--self-test") { self_test(); return 0; }
            if (arg == "--quick") quick = true;
            else if (arg == "--benchmark") benchmark = true;
            else if (arg == "--ratio-only") ratio_only = true;
            else if (arg == "--sweeps") sweeps = true;
            else if (arg == "--means") means_path = argv[++a];
            else if (arg == "--output") output = argv[++a];
            else if (arg == "--threads") threads = std::stoi(argv[++a]);
            else if (arg == "--batch-size") default_batch = std::stoi(argv[++a]);
            else if (arg == "--lower-bound") fake_lower = std::stod(argv[++a]);
            else if (arg == "--seed-base") seed_base = std::stoull(argv[++a]);
            else throw std::runtime_error("unknown argument: " + arg);
        }
        require(threads >= 1 && threads <= 64, "invalid worker count");
        require(default_batch >= 1, "invalid batch size");
        require(std::isfinite(fake_lower) && fake_lower <= 1, "invalid fake reward");
        std::ifstream input(means_path); require(bool(input), "missing MovieLens means; run prepare_data.py");
        std::vector<double> movies; std::string line;
        std::getline(input, line);
        while (std::getline(input, line)) { auto comma = line.find(','); movies.push_back(std::stod(line.substr(comma + 1))); }
        require(movies.size() >= K, "insufficient movie arms");
        std::vector<Config> tasks;
        const std::vector<long long> horizons = {10,46,215,1000,4641,21544,100000,464158,2154434,4000000,6000000,8000000,10000000};
        for (std::string learner : {"ucb", "ts"}) {
            int trials = quick ? 2 : (benchmark ? 1 : 20);
            auto add = [&](std::string grid, double x, long long horizon, double d0, int batch = 0, bool unit_range = false,
                           double lower_override = std::numeric_limits<double>::quiet_NaN()) {
                for (int trial = 0; trial < trials; ++trial) {
                    Config c; c.learner = learner; c.grid = grid; c.x = x; c.horizon = horizon;
                    c.delta0 = d0; c.trial = trial; c.batch = batch ? batch : default_batch;
                    c.lower = unit_range ? 0 : (std::isfinite(lower_override) ? lower_override : fake_lower);
                    // Pair instances and arm-wise reward streams across sweep settings and methods.
                    c.seed = seed_base + (learner == "ts" ? 1000000 : 0) + 1009ULL * trial;
                    tasks.push_back(c);
                }
            };
            double base = learner == "ucb" ? 0.2 : 1.0;
            if (benchmark) { add("benchmark",1000000,1000000,base); continue; }
            if (ratio_only) { add("ratio",100000,100000,base); continue; }
            if (!sweeps) {
                const std::vector<long long> shared_horizons = quick
                    ? std::vector<long long>{100,1000,10000}
                    : std::vector<long long>{10000,100000,200000,300000,400000,500000,
                                            600000,700000,800000,900000,1000000};
                for (long long h : shared_horizons) add("horizon",double(h),h,base);
                continue;
            }
            int points = quick ? 3 : 20;
            for (int j = 0; j < points; ++j) {
                double d0 = 0.1 + 0.4 * j / (points - 1);
                add("delta", d0, quick ? 10000 : learner == "ucb" ? 1000000 : 100000, d0);
            }
            for (long long h : quick ? std::vector<long long>{10,100,10000} : horizons) add("horizon",double(h),h,base);
            add("ratio",quick ? 10000 : 100000,quick ? 10000 : 100000,base);
            for (int batch : {1,5,10,20,50}) add("batch",batch,quick ? 10000 : 100000,base,batch);
            for (double lower : {-16.,-36.,-64.,-96.,-128.,-256.})
                add("strength",lower,quick ? 10000 : 100000,base,default_batch,false,lower);
            add("binary",0,quick ? 10000 : 100000,base,default_batch,true);
        }
        std::vector<std::array<Result, 4>> results(tasks.size());
        std::atomic<size_t> next{0}, done{0}; std::atomic<bool> failed{false};
        std::mutex log_mutex; std::string error;
        auto worker = [&]() {
            while (!failed) {
                size_t j = next.fetch_add(1); if (j >= tasks.size()) break;
                try {
                    Values means = instance(tasks[j], movies);
                    results[j][0] = simulate(tasks[j], means, "sbi");
                    results[j][1] = simulate(tasks[j], means, "pbi");
                    results[j][2] = simulate(tasks[j], means, "li");
                    results[j][3] = simulate(tasks[j], means, "heuristic", results[j][0].fake);
                    size_t count = ++done;
                    if (count % 20 == 0 || count == tasks.size()) {
                        std::lock_guard<std::mutex> lock(log_mutex);
                        std::cerr << "completed " << count << '/' << tasks.size() << " paired trials\n";
                    }
                } catch (const std::exception& e) {
                    std::lock_guard<std::mutex> lock(log_mutex); error = e.what(); failed = true;
                }
            }
        };
        std::vector<std::thread> pool;
        for (int i = 0; i < threads; ++i) pool.emplace_back(worker);
        for (auto& thread : pool) thread.join();
        require(!failed, error);
        std::filesystem::create_directories(output);
        std::ofstream raw(output + "/runs.csv"), trace(output + "/trajectories.csv");
        raw << std::setprecision(17); trace << std::setprecision(17);
        raw << "learner,grid,x,trial,seed,T,delta0,delta,sigma,r_min,r_max,batch,method,cost,fakes,target,ratio,infeasible_arms,planned_arms,recertifications";
        for (int i = 0; i < K; ++i) raw << ",mu" << i << ",fake" << i << ",genuine" << i << ",prescribed" << i;
        raw << '\n'; trace << "learner,trial,method,t,ratio\n";
        std::array<std::string, 4> names = {"sbi","pbi","li","heuristic"};
        for (size_t j = 0; j < tasks.size(); ++j) {
            const auto& c = tasks[j]; Values means = instance(c, movies);
            for (int m = 0; m < 4; ++m) {
                const auto& r = results[j][m];
                raw << c.learner << ',' << c.grid << ',' << c.x << ',' << c.trial << ',' << c.seed << ','
                    << c.horizon << ',' << c.delta0 << ',' << c.delta << ',' << c.sigma() << ',' << c.lower
                    << ',' << c.upper << ',' << c.batch << ',' << names[m] << ',' << r.cost << ','
                    << r.injected << ',' << r.target << ',' << double(r.target) / c.horizon << ','
                    << r.infeasible << ',' << r.planned << ',' << r.recertifications;
                for (int i = 0; i < K; ++i) raw << ',' << means[i] << ',' << r.fake[i] << ',' << r.genuine[i] << ',' << r.prescribed[i];
                raw << '\n';
                for (const auto& point : r.trace) trace << c.learner << ',' << c.trial << ',' << names[m] << ',' << point.first << ',' << point.second << '\n';
            }
        }
        std::cout << "Wrote " << tasks.size() * 4 << " validated runs to " << output << '\n';
        return 0;
    } catch (const std::exception& e) { std::cerr << "ERROR: " << e.what() << '\n'; return 1; }
}
