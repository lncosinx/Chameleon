/**
 * LaCAM Python bindings (pybind11).
 *
 * Exposes:
 *   lacam_solve(obstacles, starts, goals, time_limit_ms, seed, verbose)
 *     -> dict with: success, paths, makespan, sum_of_costs, runtime_ms
 */
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <lacam.hpp>
#include <fstream>
#include <sstream>
#include <cstdio>
#include <cstdlib>

namespace py = pybind11;

// ---------------------------------------------------------------------------
// Helper: write temp map file from obstacle grid
// ---------------------------------------------------------------------------

static std::string write_temp_map(const py::array_t<int>& obstacles) {
    auto buf = obstacles.request();
    int H = buf.shape[0];
    int W = buf.shape[1];
    int* ptr = static_cast<int*>(buf.ptr);

    char tmpname[] = "/tmp/lacam_map_XXXXXX";
    int fd = mkstemp(tmpname);
    if (fd < 0) throw std::runtime_error("Failed to create temp file");
    FILE* f = fdopen(fd, "w");
    fprintf(f, "height %d\nwidth %d\nmap\n", H, W);
    for (int r = 0; r < H; ++r) {
        for (int c = 0; c < W; ++c) {
            int val = ptr[r * W + c];
            // LaCAM: '.' = free, anything else ('@','T') = obstacle
            fputc(val == 0 ? '.' : '@', f);
        }
        fputc('\n', f);
    }
    fclose(f);
    return std::string(tmpname);
}

// ---------------------------------------------------------------------------
// Main solve function
// ---------------------------------------------------------------------------

py::dict lacam_solve(
    py::array_t<int> obstacles,         // (H, W) 0=free, 1=obstacle
    py::array_t<int> starts,            // (N, 2) start positions (row, col)
    py::array_t<int> goals,             // (N, 2) goal positions (row, col)
    int time_limit_ms = 10000,
    int seed = 0,
    int verbose = 0
) {
    py::dict result;

    try {
        auto obs_buf = obstacles.request();
        int H = obs_buf.shape[0];
        int W = obs_buf.shape[1];

        auto starts_buf = starts.request();
        auto goals_buf = goals.request();
        int N = starts_buf.shape[0];
        int* s_ptr = static_cast<int*>(starts_buf.ptr);
        int* g_ptr = static_cast<int*>(goals_buf.ptr);

        // Write temp map file
        std::string map_path = write_temp_map(obstacles);

        // Convert (row, col) to U indices: idx = W * r + c
        std::vector<int> start_indices(N);
        std::vector<int> goal_indices(N);
        for (int i = 0; i < N; ++i) {
            int sr = s_ptr[i * 2 + 0];
            int sc = s_ptr[i * 2 + 1];
            int gr = g_ptr[i * 2 + 0];
            int gc = g_ptr[i * 2 + 1];
            start_indices[i] = W * sr + sc;
            goal_indices[i] = W * gr + gc;
        }

        // Create instance (Graph constructed from map file internally)
        Instance ins(map_path, start_indices, goal_indices);
        std::remove(map_path.c_str());  // clean up temp file

        if (!ins.is_valid(verbose)) {
            result["success"] = false;
            result["error"] = "Invalid instance";
            return result;
        }

        // Solve
        auto MT = std::mt19937(seed);
        const auto deadline = Deadline(time_limit_ms);
        const auto solution = solve(ins, verbose, &deadline, &MT);
        const auto comp_time = deadline.elapsed_ms();

        if (solution.empty()) {
            result["success"] = false;
            result["error"] = "No solution found";
            result["runtime_ms"] = comp_time;
            return result;
        }

        // Convert solution to per-agent paths
        int makespan = (int)solution.size() - 1;
        py::list paths;
        int sum_of_costs = 0;

        for (int i = 0; i < N; ++i) {
            py::list path;
            int prev_r = -1, prev_c = -1;
            int agent_steps = 0;
            for (size_t t = 0; t < solution.size(); ++t) {
                Vertex* v = solution[t][i];
                int idx = v->index;
                int r = idx / W;
                int c = idx % W;
                path.append(py::make_tuple(r, c));
                if (t > 0 && (r != prev_r || c != prev_c)) {
                    agent_steps++;
                }
                prev_r = r; prev_c = c;
            }
            paths.append(path);
            sum_of_costs += agent_steps;
        }

        result["success"] = true;
        result["paths"] = paths;
        result["makespan"] = makespan;
        result["sum_of_costs"] = sum_of_costs;
        result["runtime_ms"] = comp_time;

    } catch (const std::exception& e) {
        result["success"] = false;
        result["error"] = std::string("Exception: ") + e.what();
    }

    return result;
}

// ---------------------------------------------------------------------------
// Module definition
// ---------------------------------------------------------------------------

PYBIND11_MODULE(_lacam, m) {
    m.doc() = "LaCAM MAPF solver Python bindings";
    m.def("solve", &lacam_solve,
          py::arg("obstacles"),
          py::arg("starts"),
          py::arg("goals"),
          py::arg("time_limit_ms") = 10000,
          py::arg("seed") = 0,
          py::arg("verbose") = 0,
          "Solve MAPF instance with LaCAM.\n\n"
          "Args:\n"
          "  obstacles: (H,W) int32 array, 0=free, 1=obstacle\n"
          "  starts: (N,2) int32 array, start positions (row,col)\n"
          "  goals: (N,2) int32 array, goal positions (row,col)\n"
          "  time_limit_ms: time limit in milliseconds\n"
          "  seed: random seed\n"
          "  verbose: verbosity level\n\n"
          "Returns: dict with success, paths, makespan, sum_of_costs, runtime_ms");
}
