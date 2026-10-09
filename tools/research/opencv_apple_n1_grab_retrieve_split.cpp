#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;

struct Stats {
    double median = 0.0;
    double p95 = 0.0;
    double p99 = 0.0;
};

static double percentile(std::vector<double> v, double q) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const double pos = q * static_cast<double>(v.size() - 1);
    const size_t lo = static_cast<size_t>(pos);
    const size_t hi = std::min(lo + 1, v.size() - 1);
    const double frac = pos - static_cast<double>(lo);
    return v[lo] * (1.0 - frac) + v[hi] * frac;
}

static Stats summarize(const std::vector<double>& v) {
    return {percentile(v, 0.50), percentile(v, 0.95), percentile(v, 0.99)};
}

static uint64_t sample(const cv::Mat& m, int frame_index) {
    if (m.empty()) return 0;
    static const int pts[6][2] = {{1,1},{17,31},{53,97},{211,503},{401,911},{701,1531}};
    const int ch = m.channels();
    uint64_t s = 0;
    for (const auto& p : pts) {
        const int y = (p[0] + frame_index * 3) % m.rows;
        const int x = (p[1] + frame_index * 5) % m.cols;
        s = s * 1315423911u + m.ptr<unsigned char>(y)[x * ch];
    }
    return s;
}

struct RunResult {
    bool ok = false;
    std::string mode;
    int frames = 0;
    int type = -1;
    int corruption = 0;
    uint64_t checksum = 0;
    Stats grab;
    Stats retrieve;
    Stats pair;
};

static RunResult run_mode(const std::string& path, bool bgra, int warmup, int measured) {
    RunResult r;
    r.mode = bgra ? "bgra" : "bgr";

    cv::VideoCapture cap(path, cv::CAP_AVFOUNDATION);
    if (!cap.isOpened()) return r;

    if (bgra) {
        const int fourcc = cv::VideoWriter::fourcc('B', 'G', 'R', 'A');
        if (!cap.set(cv::CAP_PROP_FOURCC, fourcc)) return r;
        if (static_cast<int>(cap.get(cv::CAP_PROP_FOURCC)) != fourcc) return r;
    }

    cv::Mat frame;
    for (int i = 0; i < warmup; ++i) {
        if (!cap.grab()) return r;
        if (!cap.retrieve(frame) || frame.empty()) return r;
        frame.release();
    }

    std::vector<double> grab_us;
    std::vector<double> retrieve_us;
    std::vector<double> pair_us;
    grab_us.reserve(measured);
    retrieve_us.reserve(measured);
    pair_us.reserve(measured);

    cv::Mat held;
    uint64_t held_sum = 0;
    int held_index = -1;

    for (int i = 0; i < measured; ++i) {
        const auto pair_start = Clock::now();

        const auto grab_start = Clock::now();
        const bool grabbed = cap.grab();
        const auto grab_end = Clock::now();
        if (!grabbed) break;

        const auto retrieve_start = Clock::now();
        const bool retrieved = cap.retrieve(frame);
        const auto retrieve_end = Clock::now();
        if (!retrieved || frame.empty()) break;

        const auto pair_end = Clock::now();

        grab_us.push_back(std::chrono::duration<double, std::micro>(grab_end - grab_start).count());
        retrieve_us.push_back(std::chrono::duration<double, std::micro>(retrieve_end - retrieve_start).count());
        pair_us.push_back(std::chrono::duration<double, std::micro>(pair_end - pair_start).count());

        const int logical_index = warmup + i;
        if (i == 30) {
            held = frame;
            held_sum = sample(held, logical_index);
            held_index = logical_index;
        }
        if (i == 31 && !held.empty() && sample(held, held_index) != held_sum) {
            ++r.corruption;
        }

        r.checksum ^= sample(frame, logical_index) + 0x9e3779b97f4a7c15ULL +
                      (r.checksum << 6) + (r.checksum >> 2);
        r.type = frame.type();
        ++r.frames;
        frame.release();
    }

    r.grab = summarize(grab_us);
    r.retrieve = summarize(retrieve_us);
    r.pair = summarize(pair_us);

    const int expected_type = bgra ? CV_8UC4 : CV_8UC3;
    r.ok = r.frames >= measured - 5 && r.type == expected_type && r.corruption == 0;
    return r;
}

static void print_result(const RunResult& r) {
    std::cout
        << "{\"mode\":\"" << r.mode
        << "\",\"ok\":" << (r.ok ? "true" : "false")
        << ",\"frames\":" << r.frames
        << ",\"type\":" << r.type
        << ",\"corruption\":" << r.corruption
        << ",\"checksum\":" << r.checksum
        << ",\"grab_median_us\":" << r.grab.median
        << ",\"grab_p95_us\":" << r.grab.p95
        << ",\"grab_p99_us\":" << r.grab.p99
        << ",\"retrieve_median_us\":" << r.retrieve.median
        << ",\"retrieve_p95_us\":" << r.retrieve.p95
        << ",\"retrieve_p99_us\":" << r.retrieve.p99
        << ",\"pair_median_us\":" << r.pair.median
        << ",\"pair_p95_us\":" << r.pair.p95
        << ",\"pair_p99_us\":" << r.pair.p99
        << "}\n";
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: " << argv[0] << " <video.mp4> <bgr-first|bgra-first>\n";
        return 2;
    }

    const std::string path = argv[1];
    const std::string order = argv[2];
    constexpr int warmup = 60;
    constexpr int measured = 500;

    RunResult first;
    RunResult second;
    if (order == "bgr-first") {
        first = run_mode(path, false, warmup, measured);
        second = run_mode(path, true, warmup, measured);
    } else if (order == "bgra-first") {
        first = run_mode(path, true, warmup, measured);
        second = run_mode(path, false, warmup, measured);
    } else {
        return 3;
    }

    print_result(first);
    print_result(second);

    const RunResult& bgr = first.mode == "bgr" ? first : second;
    const RunResult& bgra = first.mode == "bgra" ? first : second;

    const double retrieve_speedup =
        bgra.retrieve.median > 0.0 ? bgr.retrieve.median / bgra.retrieve.median : 0.0;
    const double grab_ratio =
        bgra.grab.median > 0.0 ? bgr.grab.median / bgra.grab.median : 0.0;
    const double pair_speedup =
        bgra.pair.median > 0.0 ? bgr.pair.median / bgra.pair.median : 0.0;

    std::cout
        << "{\"order\":\"" << order
        << "\",\"retrieve_speedup_bgr_over_bgra\":" << retrieve_speedup
        << ",\"grab_ratio_bgr_over_bgra\":" << grab_ratio
        << ",\"pair_speedup_bgr_over_bgra\":" << pair_speedup
        << "}\n";

    return (first.ok && second.ok) ? 0 : 10;
}
