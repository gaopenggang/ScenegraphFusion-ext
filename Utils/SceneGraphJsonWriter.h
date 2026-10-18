#pragma once
#include <fstream>
#include <mutex>
#include <string>
#include <Eigen/Dense>
#include <ORUtils/PathTool.hpp>
#include "../libGraphSLAM/graph.h"
#include "../libGraphSLAM/node.h"
#include "../libGraphSLAM/edge.h"

// Serialize the current scene graph to a JSON string with controlled field order:
//   top-level: frame, camera_pose, nodes, edges
//   per node:  id, label, confidence, center, bbox_max, bbox_min
//   per edge:  from, to, relation, confidence
// bbox/center stored in mm, output in metres.
// camera_pose: 4x4 camera-to-world matrix (row-major), translation in metres.
// Node confidence = latest YOLO detection score for the predicted label.
// Edge confidence = GCN relation predictor score for the predicted relation.
inline std::string SceneGraphToJson(int frame_idx, PSLAM::Graph *g,
                                    const Eigen::Matrix4f *pose = nullptr) {
    auto fmt_vec3 = [](const Eigen::Vector3f &v) -> std::string {
        char buf[96];
        std::snprintf(buf, sizeof(buf), "[%.6f,%.6f,%.6f]",
                      static_cast<double>(v[0] * 1e-3f),
                      static_cast<double>(v[1] * 1e-3f),
                      static_cast<double>(v[2] * 1e-3f));
        return buf;
    };
    auto escape = [](const std::string &s) -> std::string {
        std::string out;
        out.reserve(s.size() + 2);
        for (char c : s) {
            if      (c == '"')  out += "\\\"";
            else if (c == '\\') out += "\\\\";
            else                out += c;
        }
        return out;
    };
    auto fmt_float = [](float v) -> std::string {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%.4f", static_cast<double>(v));
        return buf;
    };

    std::string nodes_str, edges_str;
    {
        std::unique_lock<std::mutex> lock(g->mMutNode);
        bool first_node = true;
        for (const auto &kv : g->nodes) {
            const auto &n = kv.second;
            const std::string label = n->GetLabel();
            if (label == PSLAM::Node::Unknown()) continue;

            float confidence = 0.f;
            {
                std::unique_lock<std::mutex> pred_lock(n->mMutPred);
                auto it = n->mClsProb.find(label);
                if (it != n->mClsProb.end()) confidence = it->second;
            }

            // Prefer stable bbox (mBBox_min/mBBox_max, stable+valid surfels only).
            // Fall back to raw incremental bbox when not yet computed (e.g. first frame).
            Eigen::Vector3f bbox_min_out, bbox_max_out;
            {
                std::unique_lock<std::mutex> sel_lock(n->mMutSelected);
                if (n->mHasBBox) {
                    bbox_min_out = n->mBBox_min;
                    bbox_max_out = n->mBBox_max;
                } else {
                    std::unique_lock<std::mutex> node_lock(n->mMutNode);
                    if (n->bbox_min == n->bbox_max) continue; // no surfels at all
                    bbox_min_out = n->bbox_min;
                    bbox_max_out = n->bbox_max;
                }
            }

            // Only append comma after all early-continues have passed.
            if (!first_node) nodes_str += ',';
            first_node = false;

            char id_buf[32];
            std::snprintf(id_buf, sizeof(id_buf), "%d", kv.first);
            nodes_str += "{\"id\":"            + std::string(id_buf) +
                         ",\"label\":\""       + escape(label) + "\"" +
                         ",\"confidence\":"    + fmt_float(confidence) +
                         ",\"center\":"        + fmt_vec3((bbox_min_out + bbox_max_out) * 0.5f) +
                         ",\"bbox_max\":"      + fmt_vec3(bbox_max_out) +
                         ",\"bbox_min\":"      + fmt_vec3(bbox_min_out) +
                         "}";
        }
        bool first_edge = true;
        for (const auto &kv : g->edges) {
            const auto &e = kv.second;
            const std::string rel = e->GetLabel();
            if (rel == PSLAM::Edge::None()) continue;
            if (!first_edge) edges_str += ',';
            first_edge = false;

            float confidence = 0.f;
            {
                std::unique_lock<std::mutex> el(e->mMutLabel);
                auto it = e->labelProp.find(rel);
                if (it != e->labelProp.end()) confidence = it->second;
            }

            char from_buf[32], to_buf[32];
            std::snprintf(from_buf, sizeof(from_buf), "%d", e->nodeFrom);
            std::snprintf(to_buf,   sizeof(to_buf),   "%d", e->nodeTo);
            edges_str += "{\"from\":"         + std::string(from_buf) +
                         ",\"to\":"           + std::string(to_buf) +
                         ",\"relation\":\""   + escape(rel) + "\"" +
                         ",\"confidence\":"   + fmt_float(confidence) +
                         "}";
        }
    }
    // Serialize camera_pose as 4x4 row-major matrix, translation in metres.
    std::string pose_str = "null";
    if (pose) {
        char pbuf[512];
        const Eigen::Matrix4f &m = *pose;
        std::snprintf(pbuf, sizeof(pbuf),
            "[[%.6f,%.6f,%.6f,%.6f],"
             "[%.6f,%.6f,%.6f,%.6f],"
             "[%.6f,%.6f,%.6f,%.6f],"
             "[%.6f,%.6f,%.6f,%.6f]]",
            static_cast<double>(m(0,0)), static_cast<double>(m(0,1)),
            static_cast<double>(m(0,2)), static_cast<double>(m(0,3) * 1e-3f),
            static_cast<double>(m(1,0)), static_cast<double>(m(1,1)),
            static_cast<double>(m(1,2)), static_cast<double>(m(1,3) * 1e-3f),
            static_cast<double>(m(2,0)), static_cast<double>(m(2,1)),
            static_cast<double>(m(2,2)), static_cast<double>(m(2,3) * 1e-3f),
            static_cast<double>(m(3,0)), static_cast<double>(m(3,1)),
            static_cast<double>(m(3,2)), static_cast<double>(m(3,3)));
        pose_str = pbuf;
    }

    char frame_buf[32];
    std::snprintf(frame_buf, sizeof(frame_buf), "%d", frame_idx);
    return std::string("{\"frame\":") + frame_buf +
           ",\"camera_pose\":"  + pose_str +
           ",\"nodes\":["  + nodes_str  +
           "],\"edges\":["  + edges_str  + "]}";
}

inline void WriteSceneGraphJson(int frame_idx, PSLAM::Graph *g, const std::string &dir,
                                const Eigen::Matrix4f *pose = nullptr) {
    char fname[32];
    std::snprintf(fname, sizeof(fname), "frame_%06d.json", frame_idx);
    tools::PathTool::create_folder(dir);
    const std::string slash_dir = (dir.empty() || dir.back() == '/') ? dir : dir + "/";
    std::ofstream ofs(slash_dir + fname);
    ofs << SceneGraphToJson(frame_idx, g, pose);
}
