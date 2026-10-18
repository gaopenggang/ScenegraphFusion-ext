#pragma once

#include "config.h"
#include "node.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <mutex>
#include <set>
#include <string>

namespace PSLAM {

struct TeleGroundFilterContext {
    bool enabled = false;
    bool has_bounds = false;
    float scene_min_y = 0.f;
    float scene_max_y = 0.f;
};

struct TeleHorizontalPlaneNodeStats {
    Eigen::Vector3f bbox_min;
    Eigen::Vector3f bbox_max;
    float mean_normal_abs_y = 0.f;
    size_t count = 0;
};

static inline bool IsTeleGroundFilterEnabled(const ConfigPSLAM *config) {
    if (!config) return false;
    return config->pth_dataset.find("teledataset") != std::string::npos ||
           config->pth_model.find("TeleScenegraphFusion") != std::string::npos ||
           config->pth_model.find("onnx_modules") != std::string::npos;
}

static inline bool ComputeTeleSceneYBounds(const std::map<int, NodePtr> &nodes,
                                           float &scene_min_y,
                                           float &scene_max_y) {
    scene_min_y = std::numeric_limits<float>::max();
    scene_max_y = std::numeric_limits<float>::lowest();
    size_t count = 0;
    for (const auto &node_pair : nodes) {
        if (node_pair.first == 0 || !node_pair.second) continue;
        std::unique_lock<std::mutex> lock(node_pair.second->mMutSurfel);
        for (const auto &surfel_pair : node_pair.second->surfels) {
            const auto &surfel = surfel_pair.second;
            if (!surfel || !surfel->is_valid || !surfel->is_stable) continue;
            const float y = surfel->pos[1] * 1e-3f;
            scene_min_y = std::min(scene_min_y, y);
            scene_max_y = std::max(scene_max_y, y);
            ++count;
        }
    }
    return count > 0 && scene_max_y > scene_min_y;
}

static inline TeleGroundFilterContext MakeTeleGroundFilterContext(
        const ConfigPSLAM *config,
        const std::map<int, NodePtr> &nodes) {
    TeleGroundFilterContext context;
    context.enabled = IsTeleGroundFilterEnabled(config);
    if (!context.enabled) return context;
    context.has_bounds = ComputeTeleSceneYBounds(nodes, context.scene_min_y, context.scene_max_y);
    return context;
}

static inline bool ComputeTeleHorizontalPlaneNodeStats(
        const NodePtr &node,
        TeleHorizontalPlaneNodeStats &stats) {
    if (!node || node->idx == 0) return false;

    stats.bbox_min.setConstant(std::numeric_limits<float>::max());
    stats.bbox_max.setConstant(std::numeric_limits<float>::lowest());
    stats.mean_normal_abs_y = 0.f;
    stats.count = 0;
    float normal_abs_y_sum = 0.f;
    {
        std::unique_lock<std::mutex> lock(node->mMutSurfel);
        for (const auto &surfel_pair : node->surfels) {
            const auto &surfel = surfel_pair.second;
            if (!surfel || !surfel->is_valid || !surfel->is_stable) continue;
            const Eigen::Vector3f point = surfel->pos * 1e-3f;
            for (size_t d = 0; d < 3; ++d) {
                stats.bbox_min[d] = std::min(stats.bbox_min[d], point[d]);
                stats.bbox_max[d] = std::max(stats.bbox_max[d], point[d]);
            }
            normal_abs_y_sum += std::abs(surfel->normal[1]);
            ++stats.count;
        }
    }
    if (stats.count == 0) return false;
    stats.mean_normal_abs_y = normal_abs_y_sum / static_cast<float>(stats.count);
    return true;
}

static inline bool IsTeleHorizontalPlaneNode(
        const TeleHorizontalPlaneNodeStats &stats) {
    const Eigen::Vector3f dims = stats.bbox_max - stats.bbox_min;
    const float horizontal_area = std::max(dims[0], 0.f) * std::max(dims[2], 0.f);
    const float horizontal_extent = std::max(dims[0], dims[2]);
    const bool thin_y = dims[1] <= std::max(0.08f, horizontal_extent * 0.08f);
    const bool enough_area = horizontal_area >= 0.08f;
    const bool horizontal_normal = stats.mean_normal_abs_y >= 0.75f;
    return thin_y && enough_area && horizontal_normal;
}

static inline bool IsTeleGroundNode(const NodePtr &node,
                                    const TeleGroundFilterContext &context) {
    if (!context.enabled || !context.has_bounds || !node || node->idx == 0) return false;

    TeleHorizontalPlaneNodeStats stats;
    if (!ComputeTeleHorizontalPlaneNodeStats(node, stats)) return false;

    const float scene_height = std::max(context.scene_max_y - context.scene_min_y, 1e-6f);
    const float ground_band = std::min(0.25f, std::max(0.08f, scene_height * 0.08f));
    if (stats.bbox_max[1] > context.scene_min_y + ground_band) return false;

    return IsTeleHorizontalPlaneNode(stats);
}

static inline bool IsTeleCeilingNode(const NodePtr &node,
                                     const TeleGroundFilterContext &context) {
    if (!context.enabled || !context.has_bounds || !node || node->idx == 0) return false;

    TeleHorizontalPlaneNodeStats stats;
    if (!ComputeTeleHorizontalPlaneNodeStats(node, stats)) return false;

    const float scene_height = std::max(context.scene_max_y - context.scene_min_y, 1e-6f);
    const float ceiling_band = std::min(0.25f, std::max(0.08f, scene_height * 0.08f));
    if (stats.bbox_min[1] < context.scene_max_y - ceiling_band) return false;

    return IsTeleHorizontalPlaneNode(stats);
}

static inline bool IsTeleGroundSurfelLabel(unsigned short label,
                                           const std::map<int, NodePtr> &nodes,
                                           const TeleGroundFilterContext &context) {
    if (!context.enabled || !context.has_bounds || label == 0) return false;
    const auto iter = nodes.find(label);
    if (iter == nodes.end()) return false;
    return IsTeleGroundNode(iter->second, context);
}

static inline std::set<unsigned short> CollectTeleGroundLabels(
        const std::map<int, NodePtr> &nodes,
        const TeleGroundFilterContext &context) {
    std::set<unsigned short> labels;
    if (!context.enabled || !context.has_bounds) return labels;
    for (const auto &node_pair : nodes) {
        if (node_pair.first <= 0) continue;
        if (node_pair.first > std::numeric_limits<unsigned short>::max()) continue;
        if (IsTeleGroundNode(node_pair.second, context))
            labels.insert(static_cast<unsigned short>(node_pair.first));
    }
    return labels;
}

static inline std::set<unsigned short> CollectTeleCeilingLabels(
        const std::map<int, NodePtr> &nodes,
        const TeleGroundFilterContext &context) {
    std::set<unsigned short> labels;
    if (!context.enabled || !context.has_bounds) return labels;
    for (const auto &node_pair : nodes) {
        if (node_pair.first <= 0) continue;
        if (node_pair.first > std::numeric_limits<unsigned short>::max()) continue;
        if (IsTeleCeilingNode(node_pair.second, context))
            labels.insert(static_cast<unsigned short>(node_pair.first));
    }
    return labels;
}

static inline bool IsTeleGroundLabel(unsigned short label,
                                     const std::set<unsigned short> &ground_labels) {
    return ground_labels.find(label) != ground_labels.end();
}

static inline bool IsTeleCeilingLabel(unsigned short label,
                                      const std::set<unsigned short> &ceiling_labels) {
    return ceiling_labels.find(label) != ceiling_labels.end();
}

}  // namespace PSLAM
