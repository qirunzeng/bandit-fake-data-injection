// Reproduce the first UCB trigger behind the r_min=-64 scheduling rejection.
#define main simulator_entry_point
#include "simulator.cpp"
#undef main

int main() {
    std::ifstream input("experiments/data/movielens-means.csv");
    require(bool(input), "missing MovieLens means");
    std::string line;
    std::vector<double> movies;
    std::getline(input, line);
    while (std::getline(input, line)) movies.push_back(std::stod(line.substr(line.find(',') + 1)));
    std::cout << "scenario,r_min,t,arm,N_i,mean_i,N_K,mean_K,target_lower_bound,n,first_batch,post_mean,signed_gap,raw_wait,certificate,integer_wait,theorem_reward_cutoff,max_prefix_concentration_ratio\n";
    for (int scenario = 0; scenario < 4; ++scenario) {
        double lower = scenario == 0 || scenario == 3 ? -64 : scenario == 1 ? -69 : -70;
        Config c; c.learner = "ucb"; c.horizon = 1000000; c.seed = 3141592653ULL; c.lower = lower;
        Values means = instance(c, movies);
        if (scenario == 3) {
            c.constant_rewards = true;
            for (int i = 0; i < K - 1; ++i) means[i] = 0.9 - 0.05 * i;
            means[K - 1] = 0.3;
        }
        State s(c, means); s.initialize();
        long long threshold = std::ceil(std::log(double(c.horizon)) / (c.delta0 * c.delta0));
        int arm = -1;
        double concentration_ratio = 0;
        while (arm < 0) {
            for (int i = 0; i < K; ++i)
                concentration_ratio = std::max(concentration_ratio, std::abs(s.mean(i) - s.truth[i]) / s.beta(s.counts[i]));
            for (int i = 0; i < K - 1; ++i) if (s.counts[i] >= threshold) { arm = i; break; }
            if (arm < 0) s.genuine(s.choose());
        }
        long long n = s.budget(arm, s.benchmark(arm)), wait = 0;
        long long take = std::min<long long>(n, c.batch);
        double lower_target = s.mean(K - 1) - 2 * s.beta(s.counts[K - 1]);
        double post = (s.sums[arm] + take * lower) / (s.counts[arm] + take);
        double gap = lower_target - post;
        double exponent = gap * gap * (s.counts[arm] + take) / (9 * c.sigma() * c.sigma());
        bool certified = s.cooldown(arm, n, wait);
        double cutoff = s.truth[K - 1] - 3 * s.beta(1) - 3 * c.sigma() * c.delta0;
        std::cout << std::setprecision(17) << (scenario == 3 ? "deterministic" : "publication") << "," << lower << ',' << s.r.time << ',' << arm << ','
                  << s.counts[arm] << ',' << s.mean(arm) << ',' << s.counts[K - 1] << ','
                  << s.mean(K - 1) << ',' << lower_target << ',' << n << ',' << take << ','
                  << post << ',' << gap << ',' << std::exp(exponent) - s.r.time - take << ','
                  << certified << ',' << wait << ',' << cutoff << ',' << concentration_ratio << '\n';
    }
}
