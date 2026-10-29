// Exhaustive integer calibration on the declared finite experiment grid.
// Reuse the simulator without duplicating its attack or random-number logic.
#define main simulator_entry_point
#include "simulator.cpp"
#undef main

int main() {
    try {
        self_test();
        std::ifstream input("experiments/data/movielens-means.csv");
        require(bool(input), "missing MovieLens means");
        std::string line;
        std::vector<double> movies;
        std::getline(input, line);
        while (std::getline(input, line)) movies.push_back(std::stod(line.substr(line.find(',') + 1)));
        const std::vector<long long> horizons = {1000000,900000,800000,700000,600000,500000,
                                               400000,300000,200000,100000,10000};
        std::filesystem::create_directories("results/reward-search");
        std::ofstream audit("results/reward-search/candidates.csv");
        audit << "r_min,passed,tested_runs,learner,T,trial,method,stopped_at,infeasible_arms,completed_arms,post_injection_pulls,post_first_injection_pulls\n";
        for (int magnitude = 1; magnitude <= 256; ++magnitude) {
            bool passed = true;
            int tested = 0;
            Config last;
            Result result;
            std::string last_method;
            for (std::string method : {"pbi", "sbi"}) {
                for (std::string learner : {"ucb", "ts"}) {
                    for (long long horizon : horizons) {
                        for (int trial = 0; trial < 20; ++trial) {
                            Config c; c.learner = learner; c.grid = "horizon";
                            c.horizon = horizon; c.x = double(horizon); c.trial = trial;
                            c.lower = -magnitude; c.batch = 10;
                            c.delta0 = learner == "ucb" ? 0.2 : 1.0;
                            c.seed = 3141592653ULL + (learner == "ts" ? 1000000 : 0) + 1009ULL * trial;
                            result = simulate(c, instance(c, movies), method, Counts{}, true);
                            ++tested; last = c; last_method = method;
                            passed = result.time == horizon && result.infeasible == 0 &&
                                     result.completed_arms == K - 1 && result.post_injection_pulls == 0 &&
                                     result.post_first_injection_pulls == 0;
                            if (!passed) break;
                        }
                        if (!passed) break;
                    }
                    if (!passed) break;
                }
                if (!passed) break;
            }
            audit << -magnitude << ',' << passed << ',' << tested << ',' << last.learner << ','
                  << last.horizon << ',' << last.trial << ',' << last_method << ',' << result.time << ','
                  << result.infeasible << ',' << result.completed_arms << ',' << result.post_injection_pulls << ',' << result.post_first_injection_pulls << '\n';
            audit.flush();
            std::cout << "r_min=" << -magnitude << (passed ? " PASS" : " FAIL")
                      << ", checked " << tested << " runs" << std::endl;
            if (passed) {
                std::ofstream("results/reward-search/selected.txt") << -magnitude << '\n';
                return 0;
            }
        }
        throw std::runtime_error("No passing integer in the search range");
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
