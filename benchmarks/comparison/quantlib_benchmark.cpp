// QuantLib C++ Benchmark for fair comparison with SuperNova.jl
//
// Compile with:
// clang++ -std=c++17 -O3 -I$HOME/dev/QuantLib -L$HOME/dev/QuantLib/build/ql \
//         -lQuantLib -o quantlib_benchmark quantlib_benchmark.cpp
//
// Run with:
// DYLD_LIBRARY_PATH=$HOME/dev/QuantLib/build/ql ./quantlib_benchmark

#include <ql/quantlib.hpp>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <vector>
#include <numeric>
#include <cmath>

using namespace QuantLib;
using namespace std::chrono;

// Timing utility
template<typename Func>
std::pair<double, double> benchmark(Func f, int n_runs = 1000, int n_warmup = 100) {
    // Warmup
    for (int i = 0; i < n_warmup; ++i) {
        f();
    }

    // Timed runs
    std::vector<double> times;
    times.reserve(n_runs);

    for (int i = 0; i < n_runs; ++i) {
        auto start = high_resolution_clock::now();
        f();
        auto end = high_resolution_clock::now();
        double us = duration_cast<nanoseconds>(end - start).count() / 1000.0;
        times.push_back(us);
    }

    // Compute median
    std::sort(times.begin(), times.end());
    double median = times[n_runs / 2];

    // Compute mean and std
    double sum = std::accumulate(times.begin(), times.end(), 0.0);
    double mean = sum / n_runs;

    double sq_sum = 0.0;
    for (double t : times) {
        sq_sum += (t - mean) * (t - mean);
    }
    double std_dev = std::sqrt(sq_sum / n_runs);

    return {median, std_dev};
}

// European option pricing benchmark
double european_price(double S, double K, double T, double r, double sigma, Option::Type type) {
    Date today(1, January, 2025);
    Settings::instance().evaluationDate() = today;
    Date maturity = today + Integer(T * 365);

    Handle<Quote> spot(ext::make_shared<SimpleQuote>(S));
    Handle<YieldTermStructure> rTS(
        ext::make_shared<FlatForward>(today, r, Actual365Fixed()));
    Handle<YieldTermStructure> qTS(
        ext::make_shared<FlatForward>(today, 0.0, Actual365Fixed()));
    Handle<BlackVolTermStructure> volTS(
        ext::make_shared<BlackConstantVol>(today, NullCalendar(), sigma, Actual365Fixed()));

    auto process = ext::make_shared<BlackScholesMertonProcess>(spot, qTS, rTS, volTS);

    auto payoff = ext::make_shared<PlainVanillaPayoff>(type, K);
    auto exercise = ext::make_shared<EuropeanExercise>(maturity);
    VanillaOption option(payoff, exercise);

    option.setPricingEngine(ext::make_shared<AnalyticEuropeanEngine>(process));

    return option.NPV();
}

// Greeks computation
struct GreeksResult {
    double delta, gamma, vega, theta, rho;
};

GreeksResult european_greeks(double S, double K, double T, double r, double sigma, Option::Type type) {
    Date today(1, January, 2025);
    Settings::instance().evaluationDate() = today;
    Date maturity = today + Integer(T * 365);

    Handle<Quote> spot(ext::make_shared<SimpleQuote>(S));
    Handle<YieldTermStructure> rTS(
        ext::make_shared<FlatForward>(today, r, Actual365Fixed()));
    Handle<YieldTermStructure> qTS(
        ext::make_shared<FlatForward>(today, 0.0, Actual365Fixed()));
    Handle<BlackVolTermStructure> volTS(
        ext::make_shared<BlackConstantVol>(today, NullCalendar(), sigma, Actual365Fixed()));

    auto process = ext::make_shared<BlackScholesMertonProcess>(spot, qTS, rTS, volTS);

    auto payoff = ext::make_shared<PlainVanillaPayoff>(type, K);
    auto exercise = ext::make_shared<EuropeanExercise>(maturity);
    VanillaOption option(payoff, exercise);

    option.setPricingEngine(ext::make_shared<AnalyticEuropeanEngine>(process));

    return {
        option.delta(),
        option.gamma(),
        option.vega() / 100.0,   // per 1% vol
        option.theta(),
        option.rho() / 100.0     // per 1% rate
    };
}

// American option pricing (binomial tree)
double american_price(double S, double K, double T, double r, double sigma,
                      Option::Type type, int nsteps = 100) {
    Date today(1, January, 2025);
    Settings::instance().evaluationDate() = today;
    Date maturity = today + Integer(T * 365);

    Handle<Quote> spot(ext::make_shared<SimpleQuote>(S));
    Handle<YieldTermStructure> rTS(
        ext::make_shared<FlatForward>(today, r, Actual365Fixed()));
    Handle<YieldTermStructure> qTS(
        ext::make_shared<FlatForward>(today, 0.0, Actual365Fixed()));
    Handle<BlackVolTermStructure> volTS(
        ext::make_shared<BlackConstantVol>(today, NullCalendar(), sigma, Actual365Fixed()));

    auto process = ext::make_shared<BlackScholesMertonProcess>(spot, qTS, rTS, volTS);

    auto payoff = ext::make_shared<PlainVanillaPayoff>(type, K);
    auto exercise = ext::make_shared<AmericanExercise>(today, maturity);
    VanillaOption option(payoff, exercise);

    option.setPricingEngine(
        ext::make_shared<BinomialVanillaEngine<CoxRossRubinstein>>(process, nsteps));

    return option.NPV();
}

int main() {
    std::cout << std::fixed << std::setprecision(2);

    std::cout << "======================================================================\n";
    std::cout << "QUANTLIB C++ BENCHMARK\n";
    std::cout << "======================================================================\n";
    std::cout << "QuantLib version: " << QL_VERSION << "\n\n";

    // Parameters
    double S = 100.0, K = 100.0, T = 1.0, r = 0.05, sigma = 0.2;

    // European pricing benchmark
    std::cout << "----------------------------------------------------------------------\n";
    std::cout << "EUROPEAN OPTION PRICING\n";
    std::cout << "----------------------------------------------------------------------\n";

    double price = european_price(S, K, T, r, sigma, Option::Call);
    std::cout << "Price: " << std::setprecision(6) << price << "\n";

    auto [eu_median, eu_std] = benchmark([&]() {
        european_price(S, K, T, r, sigma, Option::Call);
    }, 1000, 100);

    std::cout << std::setprecision(2);
    std::cout << "Timing (1000 runs): " << eu_median << " μs (median), std=" << eu_std << " μs\n\n";

    // Greeks benchmark
    std::cout << "----------------------------------------------------------------------\n";
    std::cout << "GREEKS COMPUTATION\n";
    std::cout << "----------------------------------------------------------------------\n";

    auto greeks = european_greeks(S, K, T, r, sigma, Option::Call);
    std::cout << std::setprecision(6);
    std::cout << "Delta: " << greeks.delta << "\n";
    std::cout << "Gamma: " << greeks.gamma << "\n";
    std::cout << "Vega:  " << greeks.vega << "\n";
    std::cout << "Theta: " << greeks.theta << "\n";
    std::cout << "Rho:   " << greeks.rho << "\n";

    auto [gr_median, gr_std] = benchmark([&]() {
        european_greeks(S, K, T, r, sigma, Option::Call);
    }, 1000, 100);

    std::cout << std::setprecision(2);
    std::cout << "Timing (1000 runs): " << gr_median << " μs (median), std=" << gr_std << " μs\n\n";

    // American option benchmark
    std::cout << "----------------------------------------------------------------------\n";
    std::cout << "AMERICAN OPTION PRICING (100-step binomial)\n";
    std::cout << "----------------------------------------------------------------------\n";

    double am_price = american_price(S, K, T, r, sigma, Option::Put, 100);
    std::cout << std::setprecision(6);
    std::cout << "Price: " << am_price << "\n";

    auto [am_median, am_std] = benchmark([&]() {
        american_price(S, K, T, r, sigma, Option::Put, 100);
    }, 500, 50);

    std::cout << std::setprecision(2);
    std::cout << "Timing (500 runs): " << am_median << " μs (median), std=" << am_std << " μs\n\n";

    // Summary
    std::cout << "======================================================================\n";
    std::cout << "SUMMARY\n";
    std::cout << "======================================================================\n";
    std::cout << "  Benchmark              QuantLib C++ (μs)\n";
    std::cout << "  ─────────────────────────────────────────\n";
    std::cout << "  European pricing       " << std::setw(8) << eu_median << "\n";
    std::cout << "  Greeks (all 5)         " << std::setw(8) << gr_median << "\n";
    std::cout << "  American (100 steps)   " << std::setw(8) << am_median << "\n";
    std::cout << "======================================================================\n";

    return 0;
}
