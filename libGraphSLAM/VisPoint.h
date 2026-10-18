#pragma once
#include <Eigen/Core>
#include <array>
#include <cstdint>
#include <vector>

namespace PSLAM {
    // Lightweight visualization point: 20 bytes vs ~160 bytes for shared_ptr<Surfel>.
    // No heap allocation per point; stored as flat vector for cache-friendly iteration.
    struct VisPoint {
        Eigen::Vector3f          pos;   // world space, millimeters
        std::array<uint8_t, 3>   color; // R, G, B  (0-255)
        int                      label; // track.graph_id, used for per-object coloring
    };
    using VisPointVec = std::vector<VisPoint>;
}
