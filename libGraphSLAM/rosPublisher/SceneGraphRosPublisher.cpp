#ifdef COMPILE_WITH_ROS_PUBLISH
#include "SceneGraphRosPublisher.h"

#include <atomic>
#include <cmath>
#include <mutex>
#include <unordered_map>
#include <utility>

#include <rclcpp/rclcpp.hpp>
#include <scene_msgs/msg/tele_scene.hpp>
#include <scene_msgs/msg/tele_object.hpp>
#include <scene_msgs/msg/edge.hpp>

#include "../edge.h"
#include "../node.h"

namespace PSLAM {

namespace {

// TeleObject.ope_id values. kOpOld is never actually published (unchanged
// nodes are omitted from the delta entirely) but is defined for completeness
// and to keep numbering aligned with telescene_streamer/telescene_subscriber.
[[maybe_unused]] constexpr int32_t kOpOld = 0;
constexpr int32_t kOpAdd = 1;
constexpr int32_t kOpUpdate = 2;
constexpr int32_t kOpRemove = 3;

// A node/edge below these thresholds is considered unchanged and omitted
// from the delta message; SLAM/tracking noise would otherwise turn nearly
// every node into an UPDATE on every frame.
constexpr float kNodePositionEpsilonM = 0.10f;   // 10 cm
constexpr float kNodeSizeEpsilonM = 0.15f;       // 15 cm
constexpr float kNodeConfidenceEpsilon = 0.02f;
constexpr float kEdgeConfidenceEpsilon = 0.02f;

struct NodeSnapshot {
    std::string label;
    float confidence = 0.f;
    Eigen::Vector3f center = Eigen::Vector3f::Zero();
    Eigen::Vector3f bbox_min = Eigen::Vector3f::Zero();
    Eigen::Vector3f bbox_max = Eigen::Vector3f::Zero();
};

struct EdgeSnapshot {
    std::string relation;
    float confidence = 0.f;
};

using EdgeKey = std::pair<int, int>;
struct EdgeKeyHash {
    size_t operator()(const EdgeKey &k) const noexcept {
        return (static_cast<size_t>(static_cast<uint32_t>(k.first)) << 32) ^
               static_cast<uint32_t>(k.second);
    }
};

geometry_msgs::msg::Vector3 ToVec3(const Eigen::Vector3f &v) {
    geometry_msgs::msg::Vector3 out;
    out.x = v.x();
    out.y = v.y();
    out.z = v.z();
    return out;
}

}  // namespace

struct SceneGraphRosPublisher::Impl {
    rclcpp::Node::SharedPtr node;
    rclcpp::Publisher<scene_msgs::msg::TeleScene>::SharedPtr publisher;
    std::unordered_map<int, NodeSnapshot> prev_nodes;
    std::unordered_map<EdgeKey, EdgeSnapshot, EdgeKeyHash> prev_edges;
};

SceneGraphRosPublisher::SceneGraphRosPublisher(const std::string &topic_name)
    : mImpl(std::make_unique<Impl>()) {
    static std::atomic<int> s_instance_counter{0};
    const std::string node_name =
        "scenegraphfusion_publisher_" + std::to_string(s_instance_counter++);
    mImpl->node = std::make_shared<rclcpp::Node>(node_name);
    mImpl->publisher = mImpl->node->create_publisher<scene_msgs::msg::TeleScene>(topic_name, 10);
}

SceneGraphRosPublisher::~SceneGraphRosPublisher() = default;

void SceneGraphRosPublisher::PublishDelta(int frame_idx, Graph *g, const Eigen::Matrix4f *pose) {
    if (!g || !mImpl->publisher) return;

    // 1. Snapshot the current graph state (mm -> m, same fields/fallback
    // logic as SceneGraphJsonWriter.h so the two outputs stay consistent).
    std::unordered_map<int, NodeSnapshot> curr_nodes;
    std::unordered_map<EdgeKey, EdgeSnapshot, EdgeKeyHash> curr_edges;
    {
        std::unique_lock<std::mutex> lock(g->mMutNode);
        for (const auto &kv : g->nodes) {
            const auto &n = kv.second;
            const std::string label = n->GetLabel();
            if (label == Node::Unknown()) continue;

            float confidence = 0.f;
            {
                std::unique_lock<std::mutex> pred_lock(n->mMutPred);
                auto it = n->mClsProb.find(label);
                if (it != n->mClsProb.end()) confidence = it->second;
            }

            Eigen::Vector3f bbox_min, bbox_max;
            {
                std::unique_lock<std::mutex> sel_lock(n->mMutSelected);
                if (n->mHasBBox) {
                    bbox_min = n->mBBox_min;
                    bbox_max = n->mBBox_max;
                } else {
                    std::unique_lock<std::mutex> node_lock(n->mMutNode);
                    if (n->bbox_min == n->bbox_max) continue;  // no surfels at all
                    bbox_min = n->bbox_min;
                    bbox_max = n->bbox_max;
                }
            }

            NodeSnapshot snap;
            snap.label = label;
            snap.confidence = confidence;
            snap.bbox_min = bbox_min * 1e-3f;
            snap.bbox_max = bbox_max * 1e-3f;
            snap.center = (snap.bbox_min + snap.bbox_max) * 0.5f;
            curr_nodes.emplace(kv.first, std::move(snap));
        }

        for (const auto &kv : g->edges) {
            const auto &e = kv.second;
            const std::string rel = e->GetLabel();
            if (rel == Edge::None()) continue;

            float confidence = 0.f;
            {
                std::unique_lock<std::mutex> el(e->mMutLabel);
                auto it = e->labelProp.find(rel);
                if (it != e->labelProp.end()) confidence = it->second;
            }

            EdgeSnapshot snap;
            snap.relation = rel;
            snap.confidence = confidence;
            curr_edges.emplace(EdgeKey{e->nodeFrom, e->nodeTo}, std::move(snap));
        }
    }

    // 2. Diff against the previous frame's snapshot and build the delta message.
    scene_msgs::msg::TeleScene msg;
    msg.frame = static_cast<uint32_t>(frame_idx);
    if (pose) {
        const Eigen::Matrix4f &m = *pose;
        for (int r = 0; r < 4; ++r) {
            for (int c = 0; c < 4; ++c) {
                double v = m(r, c);
                if (c == 3 && r < 3) v *= 1e-3;  // translation: mm -> m
                msg.camera_pose[r * 4 + c] = v;
            }
        }
    }

    for (const auto &kv : curr_nodes) {
        const int id = kv.first;
        const NodeSnapshot &curr = kv.second;
        const auto prev_it = mImpl->prev_nodes.find(id);

        scene_msgs::msg::TeleObject obj;
        obj.id = static_cast<uint32_t>(id);
        obj.label = curr.label;
        obj.confidence = curr.confidence;
        obj.center = ToVec3(curr.center);
        obj.bbox_max = ToVec3(curr.bbox_max);
        obj.bbox_min = ToVec3(curr.bbox_min);

        if (prev_it == mImpl->prev_nodes.end()) {
            obj.ope_id = kOpAdd;
            obj.delta_center = ToVec3(Eigen::Vector3f::Zero());
            obj.delta_size = ToVec3(Eigen::Vector3f::Zero());
            msg.nodes.push_back(obj);
            continue;
        }

        const NodeSnapshot &prev = prev_it->second;
        const Eigen::Vector3f delta_center = curr.center - prev.center;
        const Eigen::Vector3f delta_size =
            (curr.bbox_max - curr.bbox_min) - (prev.bbox_max - prev.bbox_min);
        const bool moved = delta_center.cwiseAbs().maxCoeff() > kNodePositionEpsilonM ||
                            delta_size.cwiseAbs().maxCoeff() > kNodeSizeEpsilonM;
        const bool relabeled = curr.label != prev.label;
        const bool confidence_changed =
            std::fabs(curr.confidence - prev.confidence) > kNodeConfidenceEpsilon;

        if (moved || relabeled || confidence_changed) {
            obj.ope_id = kOpUpdate;
            obj.delta_center = ToVec3(delta_center);
            obj.delta_size = ToVec3(delta_size);
            msg.nodes.push_back(obj);
        }
    }
    for (const auto &kv : mImpl->prev_nodes) {
        if (curr_nodes.count(kv.first)) continue;  // still present, handled above
        scene_msgs::msg::TeleObject obj;
        obj.id = static_cast<uint32_t>(kv.first);
        obj.ope_id = kOpRemove;
        obj.label = kv.second.label;
        obj.confidence = 0.0;
        msg.nodes.push_back(obj);
    }

    for (const auto &kv : curr_edges) {
        const EdgeKey &key = kv.first;
        const EdgeSnapshot &curr = kv.second;
        const auto prev_it = mImpl->prev_edges.find(key);

        scene_msgs::msg::Edge edge;
        edge.start_id = static_cast<uint32_t>(key.first);
        edge.end_id = static_cast<uint32_t>(key.second);

        if (prev_it == mImpl->prev_edges.end()) {
            edge.old_relation = "";
            edge.old_confidence = 0.0;
            edge.new_relation = curr.relation;
            edge.new_confidence = curr.confidence;
            msg.edges.push_back(edge);
            continue;
        }

        const EdgeSnapshot &prev = prev_it->second;
        const bool changed = curr.relation != prev.relation ||
                              std::fabs(curr.confidence - prev.confidence) > kEdgeConfidenceEpsilon;
        if (changed) {
            edge.old_relation = prev.relation;
            edge.old_confidence = prev.confidence;
            edge.new_relation = curr.relation;
            edge.new_confidence = curr.confidence;
            msg.edges.push_back(edge);
        }
    }
    for (const auto &kv : mImpl->prev_edges) {
        if (curr_edges.count(kv.first)) continue;  // still present, handled above
        scene_msgs::msg::Edge edge;
        edge.start_id = static_cast<uint32_t>(kv.first.first);
        edge.end_id = static_cast<uint32_t>(kv.first.second);
        edge.old_relation = kv.second.relation;
        edge.old_confidence = kv.second.confidence;
        edge.new_relation = "";
        edge.new_confidence = 0.0;
        msg.edges.push_back(edge);
    }

    mImpl->prev_nodes = std::move(curr_nodes);
    mImpl->prev_edges = std::move(curr_edges);

    if (msg.nodes.empty() && msg.edges.empty()) return;  // nothing changed this frame

    msg.header.stamp = mImpl->node->get_clock()->now();
    msg.header.frame_id = "map";
    mImpl->publisher->publish(msg);
}

}  // namespace PSLAM
#endif  // COMPILE_WITH_ROS_PUBLISH
