//
// Created by sc on 8/24/20.
//
#ifdef COMPILE_WITH_ONNX
#include "GraphPredictor.h"
#include "MemoryBlock.h"
#include "OnnxModelBase.h"
#include <ORUtils/LogUtil.h>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
using namespace PSLAM;

void find_nodes_has_same_part(const std::map<int, NodePtr> &nodes, unsigned int s_idx, int t_idx, std::set<int> &has,
                              std::set<int> &exclude){
    if(nodes.find(s_idx) == nodes.end()) return;
    exclude.insert(s_idx);
    for(const auto &edge:nodes.at(s_idx)->edges) {
        if (nodes.find(edge->nodeFrom) == nodes.end()) {
            continue;
            SCLOG(ERROR) << "cannot find nodeFrom: " << edge->nodeFrom;
        }
        if (nodes.find(edge->nodeTo) == nodes.end()) {
            continue;
            SCLOG(ERROR) << "cannot find nodeFrom: " << edge->nodeTo;
        }
        if (exclude.find(edge->nodeTo) != exclude.end()) continue; // skip if excluded
        auto& nodeFrom = nodes.at(edge->nodeFrom);
        auto& nodeTo   = nodes.at(edge->nodeTo);
        const auto label_from = nodeFrom->GetLabel();
        const auto label_to = nodeTo->GetLabel();
        if (label_from == Node::Unknown() || label_to == Node::Unknown() ||
            label_from != label_to) continue;
        if (nodeFrom->neighbors.find(edge->nodeTo) == nodeFrom->neighbors.end()) continue; // if target node is not the neighbor of the source node
        if(!nodeFrom->CheckConnectivity(nodeTo.get(),5,false)) continue; // if they are not connected, continue
        if (edge->GetLabel() == Edge::Same()) {
            has.insert(edge->nodeTo);
            find_nodes_has_same_part(nodes,edge->nodeTo,t_idx,has,exclude);
        }
    }
}

static MemoryBlock2D ConcatObjFeature(size_t n_node, size_t dim_obj_feature, Ort::Value &obj_feature,
                                      const MemoryBlock2D &descriptor) {
    /// Concat descriptor to the object feature
    MemoryBlock2D out(MemoryBlock::DATA_TYPE::FLOAT, {static_cast<unsigned long>(n_node), dim_obj_feature + 8});
    MemoryBlock2D tmp_out_enc_obj(obj_feature.GetTensorMutableData<float>(), MemoryBlock::FLOAT,
                                  {static_cast<unsigned long>(n_node), dim_obj_feature});
    for (size_t n = 0; n < n_node; ++n) {
        MemoryBlock tmp (descriptor.Get<float>(n,3),descriptor.type(),8);
//                SCLOG(VERBOSE) << tmp;
        size_t offset= 0;
        memcpy(out.Get<float>(n, offset), tmp_out_enc_obj.Get<float>(n,0), sizeof(float)*dim_obj_feature);
        offset += dim_obj_feature;
        memcpy(out.Get<float>(n, offset), tmp.Get<float>(), sizeof(float)*6);
        offset += 6;
        out.at<float>(n, offset) = std::log( tmp.at<float>(6) );
        offset+=1;
        out.at<float>(n, offset) = std::log( tmp.at<float>(7) );
    }
    return out;
}

void calculateBBox(const std::vector<float> &points, float *min, float *max){
    assert(points.size()%3==0);
    for(size_t i=0;i<points.size()/3;++i){
        for(size_t j=0;j<3;++j){
            if(points[3*i+j]<min[j]) min[j]=points[3*i+j];
            if(points[3*i+j]>max[j]) max[j]=points[3*i+j];
        }
    }
}

static std::string WithTrailingSlash(std::string path) {
    if (!path.empty() && path.back() != '/') path += "/";
    return path;
}

static std::string ParentDir(std::string path) {
    while (!path.empty() && path.back() == '/') path.pop_back();
    const auto pos = path.find_last_of('/');
    if (pos == std::string::npos) return "";
    return path.substr(0, pos + 1);
}

static bool FileExists(const std::string &path) {
    std::ifstream file(path, std::ios::in);
    return file.good();
}

static bool ReadDepthMillimeters(const cv::Mat &depth, int y, int x, float &depth_mm) {
    if (depth.empty() || depth.channels() != 1) return false;
    switch (depth.depth()) {
        case CV_16U:
            depth_mm = static_cast<float>(depth.at<unsigned short>(y, x));
            break;
        case CV_16S:
            depth_mm = static_cast<float>(depth.at<short>(y, x));
            break;
        case CV_32S:
            depth_mm = static_cast<float>(depth.at<int>(y, x));
            break;
        case CV_32F:
            depth_mm = depth.at<float>(y, x);
            break;
        case CV_64F:
            depth_mm = static_cast<float>(depth.at<double>(y, x));
            break;
        default:
            return false;
    }
    return std::isfinite(depth_mm) && depth_mm > 1e-6f;
}

static bool IsTeleOnnxModuleDirExact(const std::string &path) {
    const auto folder = WithTrailingSlash(path);
    return FileExists(folder + "obj_encoder.onnx") &&
           FileExists(folder + "node_geometry.onnx") &&
           FileExists(folder + "node_feature_fuser.onnx") &&
           FileExists(folder + "edge_encoder.onnx") &&
           FileExists(folder + "gnn.onnx") &&
           FileExists(folder + "rel_predictor.onnx");
}

static std::string ResolveTeleOnnxModuleDir(const std::string &path) {
    const auto folder = WithTrailingSlash(path);
    if (IsTeleOnnxModuleDirExact(folder)) return folder;
    const auto child = folder + "onnx_modules/";
    if (IsTeleOnnxModuleDirExact(child)) return child;
    return folder;
}

static bool IsTeleOnnxModuleDir(const std::string &path) {
    return IsTeleOnnxModuleDirExact(ResolveTeleOnnxModuleDir(path));
}

static bool HasTeleModule(const std::string &path, const std::string &name) {
    return FileExists(ResolveTeleOnnxModuleDir(path) + name + ".onnx");
}

static std::string FindTeleVisualEncoderPath(const std::string &path_onnx) {
    const std::vector<std::string> candidates = {
            WithTrailingSlash(path_onnx) + "visual_encoder.onnx",
            "/home/gao/codes/SceneGraphFusion/dinov2/visual_encoder.onnx"
    };
    for (const auto &path : candidates)
        if (FileExists(path)) return path;
    return "";
}

static std::vector<std::string> ReadNonEmptyLines(const std::vector<std::string> &paths,
                                                  const std::string &what) {
    for (const auto &path : paths) {
        std::fstream file(path, std::ios::in);
        if (!file.is_open()) continue;
        std::vector<std::string> lines;
        std::string line;
        while (std::getline(file, line)) {
            while (!line.empty() && (line.back() == '\r' || line.back() == '\n' || line.back() == ' '))
                line.pop_back();
            if (!line.empty()) lines.emplace_back(line);
        }
        if (!lines.empty()) {
            SCLOG(INFO) << "Loaded " << what << " from " << path;
            return lines;
        }
    }
    throw std::runtime_error("Cannot find " + what + " file.");
}

static MemoryBlock2D CopyTensor2D(Ort::Value &tensor) {
    auto shape = tensor.GetTensorTypeAndShapeInfo().GetShape();
    if (shape.size() != 2) throw std::runtime_error("Expected a 2D tensor.");
    if (shape[0] < 0 || shape[1] < 0) throw std::runtime_error("Tensor has unresolved dynamic shape.");
    MemoryBlock2D out(MemoryBlock::FLOAT,
                      {static_cast<size_t>(shape[0]), static_cast<size_t>(shape[1])});
    if (out.size() > 0)
        std::memcpy(out.Get<float>(), tensor.GetTensorMutableData<float>(), sizeof(float) * out.size());
    return out;
}

static float Sigmoid(float value) {
    return 1.f / (1.f + std::exp(-value));
}

static void SoftmaxRow(const float *logits, size_t dim, std::vector<float> &out) {
    out.resize(dim);
    if (dim == 0) return;
    float max_value = logits[0];
    for (size_t i = 1; i < dim; ++i) max_value = std::max(max_value, logits[i]);
    float sum = 0.f;
    for (size_t i = 0; i < dim; ++i) {
        out[i] = std::exp(logits[i] - max_value);
        sum += out[i];
    }
    if (sum <= std::numeric_limits<float>::epsilon()) return;
    for (size_t i = 0; i < dim; ++i) out[i] /= sum;
}

static bool IsTeleLargeObjectLabel(const std::string &label) {
    static const std::unordered_set<std::string> labels = {
            "beanbag",
            "cabinet",
            "sofa",
            "table",
            "tvstand",
            "wall_cabinet",
            "chest_of_drawers",
            "fridge",
            "kitchen_counter",
            "kitchen_cupboard"
    };
    return labels.find(label) != labels.end();
}

static bool ShouldSuppressTeleSparseLargeObject(const std::string &label,
                                                size_t stable_surfel_count,
                                                int segment_filter) {
    const size_t min_count = std::max<size_t>(30, static_cast<size_t>(std::max(1, segment_filter)) * 3);
    return IsTeleLargeObjectLabel(label) && stable_surfel_count < min_count;
}

static bool IsTeleGroundOnlyObjectLabel(const std::string &label) {
    static const std::unordered_set<std::string> labels = {
            "beanbag",
            "bike",
            "cabinet",
            "chair",
            "indoor_plant",
            "sofa",
            "table",
            "tvstand",
            "wall_cabinet",
            "chest_of_drawers",
            "door",
            "fridge",
            "kitchen_counter"
    };
    return labels.find(label) != labels.end();
}

static bool IsTeleAirOnlyObjectLabel(const std::string &label) {
    static const std::unordered_set<std::string> labels = {
            "clothes_hanger",
            "tv_screen",
            "kitchen_cupboard"
    };
    return labels.find(label) != labels.end();
}

static bool IsTeleRareObjectLabel(const std::string &label) {
    static const std::unordered_set<std::string> labels = {
            "bike",
            "door",
            "clothes_hanger",
            "indoor_plant"
    };
    return labels.find(label) != labels.end();
}

static std::string TeleLabelName(const std::map<size_t, std::string> &labels, size_t idx) {
    const auto iter = labels.find(idx);
    return iter != labels.end() ? iter->second : Node::Unknown();
}

static std::vector<size_t> TopKIndices(const std::vector<float> &prob, size_t k) {
    std::vector<size_t> order(prob.size());
    std::iota(order.begin(), order.end(), 0);
    const size_t top_k = std::min(k, order.size());
    std::partial_sort(order.begin(), order.begin() + top_k, order.end(),
                      [&](size_t a, size_t b) { return prob.at(a) > prob.at(b); });
    order.resize(top_k);
    return order;
}

static bool IsTeleNodeSuspended(const Eigen::Vector3f &bbox_min,
                                float scene_min_y,
                                float scene_max_y) {
    const float scene_height = scene_max_y - scene_min_y;
    if (scene_height <= 1e-6f) return false;
    const float clearance = bbox_min[1] - scene_min_y;
    const float threshold = std::max(0.25f, scene_height * 0.12f);
    return clearance > threshold;
}

struct TeleNodeLabelDecision {
    size_t label_idx = 0;
    bool low_confidence = false;
    bool force_update = false;
    bool skip_prediction = false;
    std::string reason = "top1";
};

static TeleNodeLabelDecision SelectTeleNodeLabelWithPriors(
        const std::vector<float> &prob,
        const std::map<size_t, std::string> &labels,
        const Eigen::Vector3f &bbox_min,
        float scene_min_y,
        float scene_max_y,
        float no_prediction_threshold,
        float prior_confidence_threshold,
        int prior_top_k) {
    TeleNodeLabelDecision decision;
    const auto top_candidates = TopKIndices(prob, std::max(1, prior_top_k));
    if (top_candidates.empty()) return decision;

    decision.label_idx = top_candidates.front();
    const float top_confidence = prob.at(top_candidates.front());
    const float skip_threshold = std::max(0.0f, no_prediction_threshold);
    const float prior_threshold = std::max(skip_threshold, prior_confidence_threshold);
    if (top_confidence < skip_threshold) {
        decision.low_confidence = true;
        decision.skip_prediction = true;
        decision.reason = "below_no_prediction_threshold";
        return decision;
    }

    if (top_confidence >= prior_threshold) return decision;

    decision.low_confidence = true;
    const bool suspended = IsTeleNodeSuspended(bbox_min, scene_min_y, scene_max_y);
    const size_t invalid_idx = std::numeric_limits<size_t>::max();
    size_t best_compatible_idx = invalid_idx;
    size_t best_non_rare_compatible_idx = invalid_idx;
    for (const size_t idx : top_candidates) {
        const std::string label = TeleLabelName(labels, idx);
        const bool compatible = suspended ?
                                IsTeleAirOnlyObjectLabel(label) :
                                IsTeleGroundOnlyObjectLabel(label);
        if (!compatible) continue;
        if (best_compatible_idx == invalid_idx)
            best_compatible_idx = idx;
        if (!IsTeleRareObjectLabel(label)) {
            best_non_rare_compatible_idx = idx;
            break;
        }
    }

    const size_t selected_idx = best_non_rare_compatible_idx != invalid_idx ?
                                best_non_rare_compatible_idx : best_compatible_idx;
    if (selected_idx != invalid_idx) {
        decision.label_idx = selected_idx;
        decision.force_update = true;
        const bool selected_top1 = selected_idx == top_candidates.front();
        const bool demoted_rare = best_compatible_idx != invalid_idx &&
                                  best_non_rare_compatible_idx != invalid_idx &&
                                  best_compatible_idx != best_non_rare_compatible_idx &&
                                  IsTeleRareObjectLabel(TeleLabelName(labels, best_compatible_idx));
        const bool rare_only = best_non_rare_compatible_idx == invalid_idx &&
                               IsTeleRareObjectLabel(TeleLabelName(labels, selected_idx));
        if (demoted_rare) {
            decision.reason = suspended ? "air_prior_rare_demoted_topk" :
                              "ground_prior_rare_demoted_topk";
        } else if (rare_only) {
            decision.reason = suspended ?
                              (selected_top1 ? "air_prior_rare_top1" : "air_prior_rare_topk") :
                              (selected_top1 ? "ground_prior_rare_top1" : "ground_prior_rare_topk");
        } else {
            decision.reason = suspended ?
                              (selected_top1 ? "air_prior_top1" : "air_prior_topk") :
                              (selected_top1 ? "ground_prior_top1" : "ground_prior_topk");
        }
        return decision;
    }

    decision.reason = suspended ? "low_conf_air_no_match" : "low_conf_ground_no_match";
    return decision;
}

static size_t ProjectSurfelBBox(const NodePtr &node,
                                const Eigen::Matrix4f &world_to_camera,
                                const CameraParameters &cam,
                                float &x1, float &y1, float &x2, float &y2) {
    x1 = std::numeric_limits<float>::max();
    y1 = std::numeric_limits<float>::max();
    x2 = std::numeric_limits<float>::lowest();
    y2 = std::numeric_limits<float>::lowest();
    size_t valid = 0;

    std::vector<SurfelPtr> surfels;
    {
        std::unique_lock<std::mutex> lock(node->mMutSelected);
        surfels = node->selected_surfels;
    }
    if (surfels.empty()) {
        std::unique_lock<std::mutex> lock(node->mMutSurfel);
        for (const auto &pair : node->surfels)
            if (pair.second->is_stable && pair.second->is_valid)
                surfels.emplace_back(pair.second);
    }

    for (const auto &surfel : surfels) {
        Eigen::Vector4f point;
        point << surfel->pos[0], surfel->pos[1], surfel->pos[2], 1.f;
        const Eigen::Vector4f camera_point = world_to_camera * point;
        const float z = camera_point[2];
        if (z <= 1e-4f) continue;
        const float u = cam.fx * (camera_point[0] / z) + cam.cx;
        const float v = cam.fy * (camera_point[1] / z) + cam.cy;
        if (u < 0.f || v < 0.f || u >= cam.width || v >= cam.height) continue;
        x1 = std::min(x1, u);
        y1 = std::min(y1, v);
        x2 = std::max(x2, u);
        y2 = std::max(y2, v);
        ++valid;
    }
    return valid;
}

static cv::Vec3b YoloClassColor(int class_idx) {
    float h = fmodf(float(class_idx + 1) * 0.618033988749895f, 1.0f);
    float c = 0.8f * 0.95f;
    float x = c * (1.f - fabsf(fmodf(h * 6.f, 2.f) - 1.f));
    float m = 0.95f - c;
    float r, g, b;
    switch (int(h * 6.f) % 6) {
        case 0: r=c; g=x; b=0; break;
        case 1: r=x; g=c; b=0; break;
        case 2: r=0; g=c; b=x; break;
        case 3: r=0; g=x; b=c; break;
        case 4: r=x; g=0; b=c; break;
        default: r=c; g=0; b=x; break;
    }
    return {uint8_t((b+m)*255), uint8_t((g+m)*255), uint8_t((r+m)*255)}; // BGR
}

// TeleYoloDetection is defined in GraphPredictor.h (inside COMPILE_WITH_ONNX).

struct TeleYoloAssignment {
    int class_idx = -1;
    float score = 0.f;
    float overlap = 0.f;
};

struct TeleLetterboxInfo {
    float scale = 1.f;
    int pad_x = 0;
    int pad_y = 0;
    int resized_w = 0;
    int resized_h = 0;
};

static float RectIoU(const cv::Rect &a, const cv::Rect &b) {
    const int x1 = std::max(a.x, b.x);
    const int y1 = std::max(a.y, b.y);
    const int x2 = std::min(a.x + a.width, b.x + b.width);
    const int y2 = std::min(a.y + a.height, b.y + b.height);
    const int inter_w = std::max(0, x2 - x1);
    const int inter_h = std::max(0, y2 - y1);
    const float inter = static_cast<float>(inter_w * inter_h);
    const float area_a = static_cast<float>(std::max(0, a.width) * std::max(0, a.height));
    const float area_b = static_cast<float>(std::max(0, b.width) * std::max(0, b.height));
    const float denom = area_a + area_b - inter;
    return denom <= 1e-6f ? 0.f : inter / denom;
}

static cv::Mat LetterboxToTensorRGB(const cv::Mat &rgb_image,
                                    int input_w,
                                    int input_h,
                                    TeleLetterboxInfo &info,
                                    std::vector<float> &tensor) {
    info.scale = std::min(static_cast<float>(input_w) / static_cast<float>(rgb_image.cols),
                          static_cast<float>(input_h) / static_cast<float>(rgb_image.rows));
    info.resized_w = std::max(1, static_cast<int>(std::round(rgb_image.cols * info.scale)));
    info.resized_h = std::max(1, static_cast<int>(std::round(rgb_image.rows * info.scale)));
    info.pad_x = (input_w - info.resized_w) / 2;
    info.pad_y = (input_h - info.resized_h) / 2;

    cv::Mat resized;
    cv::resize(rgb_image, resized, cv::Size(info.resized_w, info.resized_h), 0, 0, cv::INTER_LINEAR);
    cv::Mat canvas(input_h, input_w, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(canvas(cv::Rect(info.pad_x, info.pad_y, info.resized_w, info.resized_h)));

    tensor.assign(static_cast<size_t>(3 * input_w * input_h), 0.f);
    for (int y = 0; y < input_h; ++y) {
        const auto *row = canvas.ptr<cv::Vec3b>(y);
        for (int x = 0; x < input_w; ++x) {
            const size_t p = static_cast<size_t>(y * input_w + x);
            tensor[p] = static_cast<float>(row[x][0]) / 255.f;
            tensor[static_cast<size_t>(input_w * input_h) + p] =
                    static_cast<float>(row[x][1]) / 255.f;
            tensor[static_cast<size_t>(2 * input_w * input_h) + p] =
                    static_cast<float>(row[x][2]) / 255.f;
        }
    }
    return canvas;
}

static cv::Rect BoxFromYolo(float cx, float cy, float w, float h,
                            const TeleLetterboxInfo &letterbox,
                            int image_w,
                            int image_h) {
    float x1 = (cx - w * 0.5f - static_cast<float>(letterbox.pad_x)) / letterbox.scale;
    float y1 = (cy - h * 0.5f - static_cast<float>(letterbox.pad_y)) / letterbox.scale;
    float x2 = (cx + w * 0.5f - static_cast<float>(letterbox.pad_x)) / letterbox.scale;
    float y2 = (cy + h * 0.5f - static_cast<float>(letterbox.pad_y)) / letterbox.scale;
    x1 = std::max(0.f, std::min(static_cast<float>(image_w), x1));
    y1 = std::max(0.f, std::min(static_cast<float>(image_h), y1));
    x2 = std::max(0.f, std::min(static_cast<float>(image_w), x2));
    y2 = std::max(0.f, std::min(static_cast<float>(image_h), y2));
    const int left = static_cast<int>(std::floor(x1));
    const int top = static_cast<int>(std::floor(y1));
    const int right = static_cast<int>(std::ceil(x2));
    const int bottom = static_cast<int>(std::ceil(y2));
    return cv::Rect(left, top, std::max(0, right - left), std::max(0, bottom - top));
}

static cv::Mat DecodeYoloMask(const float *proto,
                              const float *coeff,
                              size_t mask_dim,
                              int proto_h,
                              int proto_w,
                              const TeleLetterboxInfo &letterbox,
                              const cv::Rect &box,
                              int image_w,
                              int image_h,
                              int input_w,
                              int input_h) {
    cv::Mat mask_small(proto_h, proto_w, CV_32F, cv::Scalar(0));
    const size_t proto_area = static_cast<size_t>(proto_h * proto_w);
    for (size_t k = 0; k < mask_dim; ++k) {
        const float c = coeff[k];
        const float *src = proto + k * proto_area;
        float *dst = mask_small.ptr<float>(0);
        for (size_t p = 0; p < proto_area; ++p)
            dst[p] += c * src[p];
    }
    for (int y = 0; y < proto_h; ++y) {
        float *row = mask_small.ptr<float>(y);
        for (int x = 0; x < proto_w; ++x)
            row[x] = Sigmoid(row[x]);
    }

    cv::Mat mask_input;
    cv::resize(mask_small, mask_input, cv::Size(input_w, input_h), 0, 0, cv::INTER_LINEAR);
    cv::Rect unpadded(letterbox.pad_x, letterbox.pad_y,
                      letterbox.resized_w, letterbox.resized_h);
    unpadded &= cv::Rect(0, 0, input_w, input_h);
    cv::Mat mask_unpadded = mask_input(unpadded).clone();
    cv::Mat mask_orig_float;
    cv::resize(mask_unpadded, mask_orig_float, cv::Size(image_w, image_h), 0, 0, cv::INTER_LINEAR);
    cv::Mat mask_orig;
    cv::threshold(mask_orig_float, mask_orig, 0.5, 255.0, cv::THRESH_BINARY);
    mask_orig.convertTo(mask_orig, CV_8U);
    if (box.area() > 0) {
        cv::Mat cropped = cv::Mat::zeros(mask_orig.size(), CV_8U);
        cv::Rect clipped = box & cv::Rect(0, 0, image_w, image_h);
        if (clipped.area() > 0)
            mask_orig(clipped).copyTo(cropped(clipped));
        return cropped;
    }
    return mask_orig;
}

static std::vector<TeleYoloDetection> RunTeleYoloSegmentation(
        Ort::Session &session,
        const Ort::MemoryInfo *memory_info,
        const cv::Mat &rgb_image,
        int num_classes,
        float conf_threshold,
        float iou_threshold,
        int min_mask_area) {
    std::vector<TeleYoloDetection> detections;
    if (rgb_image.empty() || rgb_image.channels() != 3 || num_classes <= 0) return detections;

    const int input_w = 640;
    const int input_h = 640;
    TeleLetterboxInfo letterbox;
    std::vector<float> input_tensor;
    LetterboxToTensorRGB(rgb_image, input_w, input_h, letterbox, input_tensor);

    std::vector<int64_t> input_shape = {1, 3, input_h, input_w};
    auto input = ONNX::CreateTensor(memory_info, input_tensor.data(), input_shape);
    const char *input_names[] = {"images"};
    const char *output_names[] = {"output0", "output1"};
    auto outputs = session.Run(Ort::RunOptions{nullptr},
                               input_names, &input, 1,
                               output_names, 2);
    auto out0_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    auto out1_shape = outputs[1].GetTensorTypeAndShapeInfo().GetShape();
    if (out0_shape.size() != 3 || out1_shape.size() != 4)
        return detections;
    const int channels = static_cast<int>(out0_shape[1]);
    const int anchors = static_cast<int>(out0_shape[2]);
    const int mask_dim = channels - 4 - num_classes;
    if (mask_dim <= 0) return detections;
    const int proto_c = static_cast<int>(out1_shape[1]);
    const int proto_h = static_cast<int>(out1_shape[2]);
    const int proto_w = static_cast<int>(out1_shape[3]);
    if (proto_c < mask_dim || proto_h <= 0 || proto_w <= 0) return detections;

    const float *pred = outputs[0].GetTensorMutableData<float>();
    const float *proto = outputs[1].GetTensorMutableData<float>();

    struct Candidate {
        cv::Rect box;
        int class_idx = -1;
        float score = 0.f;
        std::vector<float> coeff;
    };
    std::vector<Candidate> candidates;
    for (int a = 0; a < anchors; ++a) {
        int best_class = -1;
        float best_score = 0.f;
        for (int c = 0; c < num_classes; ++c) {
            const float score = pred[(4 + c) * anchors + a];
            if (score > best_score) {
                best_score = score;
                best_class = c;
            }
        }
        if (best_score < conf_threshold || best_class < 0) continue;
        const float cx = pred[0 * anchors + a];
        const float cy = pred[1 * anchors + a];
        const float w = pred[2 * anchors + a];
        const float h = pred[3 * anchors + a];
        cv::Rect box = BoxFromYolo(cx, cy, w, h, letterbox,
                                   rgb_image.cols, rgb_image.rows);
        if (box.area() <= 0) continue;
        Candidate cand;
        cand.box = box;
        cand.class_idx = best_class;
        cand.score = best_score;
        cand.coeff.resize(static_cast<size_t>(mask_dim));
        for (int k = 0; k < mask_dim; ++k)
            cand.coeff[static_cast<size_t>(k)] = pred[(4 + num_classes + k) * anchors + a];
        candidates.emplace_back(std::move(cand));
    }

    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate &a, const Candidate &b) { return a.score > b.score; });
    std::vector<Candidate> kept;
    std::vector<char> removed(candidates.size(), 0);
    for (size_t i = 0; i < candidates.size(); ++i) {
        if (removed[i]) continue;
        kept.emplace_back(candidates[i]);
        for (size_t j = i + 1; j < candidates.size(); ++j) {
            if (removed[j]) continue;
            if (candidates[i].class_idx != candidates[j].class_idx) continue;
            if (RectIoU(candidates[i].box, candidates[j].box) > iou_threshold)
                removed[j] = 1;
        }
    }

    for (const auto &cand : kept) {
        cv::Mat mask = DecodeYoloMask(proto,
                                      cand.coeff.data(),
                                      static_cast<size_t>(mask_dim),
                                      proto_h, proto_w,
                                      letterbox, cand.box,
                                      rgb_image.cols, rgb_image.rows,
                                      input_w, input_h);
        if (cv::countNonZero(mask) < min_mask_area) continue;
        TeleYoloDetection det;
        det.box = cand.box;
        det.mask = std::move(mask);
        det.class_idx = cand.class_idx;
        det.score = cand.score;
        detections.emplace_back(std::move(det));
    }
    return detections;
}

static std::vector<SurfelPtr> CollectNodeSurfelsForProjection(const NodePtr &node) {
    std::vector<SurfelPtr> surfels;
    {
        std::unique_lock<std::mutex> lock(node->mMutSelected);
        surfels = node->selected_surfels;
    }
    if (!surfels.empty()) return surfels;
    {
        std::unique_lock<std::mutex> lock(node->mMutSurfel);
        surfels.reserve(node->surfels.size());
        for (const auto &pair : node->surfels)
            if (pair.second->is_stable && pair.second->is_valid)
                surfels.emplace_back(pair.second);
    }
    return surfels;
}

static TeleYoloAssignment AssignNodeFromYoloWithPose(
        const NodePtr &node,
        const std::vector<TeleYoloDetection> &detections,
        const Eigen::Matrix4f &world_to_camera,
        const CameraParameters &cam,
        float overlap_threshold) {
    TeleYoloAssignment assignment;
    if (detections.empty()) return assignment;

    const auto surfels = CollectNodeSurfelsForProjection(node);
    if (surfels.empty()) return assignment;
    std::vector<int> overlap(detections.size(), 0);
    int visible = 0;
    for (const auto &surfel : surfels) {
        Eigen::Vector4f point;
        point << surfel->pos[0], surfel->pos[1], surfel->pos[2], 1.f;
        const Eigen::Vector4f camera_point = world_to_camera * point;
        const float z = camera_point[2];
        if (z <= 1e-4f) continue;
        const int u = static_cast<int>(std::round(cam.fx * (camera_point[0] / z) + cam.cx));
        const int v = static_cast<int>(std::round(cam.fy * (camera_point[1] / z) + cam.cy));
        if (u < 0 || v < 0 || u >= cam.width || v >= cam.height) continue;
        ++visible;
        for (size_t d = 0; d < detections.size(); ++d) {
            const auto &mask = detections[d].mask;
            if (mask.empty()) continue;
            if (mask.at<unsigned char>(v, u) > 0) ++overlap[d];
        }
    }
    if (visible <= 0) return assignment;

    int best_det = -1;
    float best_overlap = 0.f;
    float best_score = 0.f;
    for (size_t d = 0; d < detections.size(); ++d) {
        const float ratio = static_cast<float>(overlap[d]) / static_cast<float>(visible);
        if (ratio > best_overlap || (std::abs(ratio - best_overlap) < 1e-6f &&
                                     detections[d].score > best_score)) {
            best_overlap = ratio;
            best_score = detections[d].score;
            best_det = static_cast<int>(d);
        }
    }
    if (best_det >= 0 && best_overlap >= overlap_threshold) {
        assignment.class_idx = detections[static_cast<size_t>(best_det)].class_idx;
        assignment.score = detections[static_cast<size_t>(best_det)].score;
        assignment.overlap = best_overlap;
    }
    return assignment;
}

static TeleYoloAssignment AssignNodeFromYolo(
        const NodePtr &node,
        const std::vector<TeleYoloDetection> &detections,
        const Eigen::Matrix4f &pose,
        const CameraParameters &cam,
        float overlap_threshold) {
    const auto a = AssignNodeFromYoloWithPose(node, detections, pose, cam, overlap_threshold);
    const auto b = AssignNodeFromYoloWithPose(node, detections, pose.inverse(), cam, overlap_threshold);
    if (b.class_idx >= 0 && (a.class_idx < 0 || b.overlap > a.overlap)) return b;
    return a;
}

struct TeleYoloObjectObservation {
    int class_idx = -1;
    float score = 0.f;
    Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
    Eigen::Vector3f bbox_min = Eigen::Vector3f::Zero();
    Eigen::Vector3f bbox_max = Eigen::Vector3f::Zero();
    std::vector<SurfelPtr> surfels;
};

static std::string TeleVoxelKey(const Eigen::Vector3f &point_mm, float voxel_m) {
    const float voxel_mm = std::max(1.0f, voxel_m * 1000.0f);
    const int x = static_cast<int>(std::floor(point_mm[0] / voxel_mm));
    const int y = static_cast<int>(std::floor(point_mm[1] / voxel_mm));
    const int z = static_cast<int>(std::floor(point_mm[2] / voxel_mm));
    return std::to_string(x) + "_" + std::to_string(y) + "_" + std::to_string(z);
}

static bool BuildTeleYoloObjectObservation(
        const TeleYoloDetection &det,
        const cv::Mat &rgb,
        const cv::Mat &depth,
        const Eigen::Matrix4f &world_to_camera,
        const CameraParameters &cam,
        const ConfigPSLAM *config,
        TeleYoloObjectObservation &observation) {
    observation = TeleYoloObjectObservation{};
    if (det.mask.empty() || depth.empty() || rgb.empty()) return false;
    if (cam.fx <= 1e-6f || cam.fy <= 1e-6f) return false;

    const int stride = std::max(1, config ? config->tele_yolo_object_sample_stride : 2);
    const int max_points = std::max(1, config ? config->tele_yolo_object_max_points_per_mask : 3000);
    const Eigen::Matrix4f camera_to_world = world_to_camera.inverse();

    Eigen::Vector3f sum = Eigen::Vector3f::Zero();
    Eigen::Vector3f bbox_min, bbox_max;
    bbox_min.setConstant(std::numeric_limits<float>::max());
    bbox_max.setConstant(std::numeric_limits<float>::lowest());
    std::vector<SurfelPtr> surfels;
    surfels.reserve(static_cast<size_t>(max_points));

    const cv::Rect bounds = det.box & cv::Rect(0, 0, depth.cols, depth.rows);
    for (int v = bounds.y; v < bounds.y + bounds.height && surfels.size() < static_cast<size_t>(max_points); v += stride) {
        for (int u = bounds.x; u < bounds.x + bounds.width && surfels.size() < static_cast<size_t>(max_points); u += stride) {
            if (det.mask.at<unsigned char>(v, u) == 0) continue;
            float depth_mm = 0.f;
            if (!ReadDepthMillimeters(depth, v, u, depth_mm)) continue;
            if (depth_mm < 1e-3f) continue;

            Eigen::Vector4f camera_point;
            camera_point << (static_cast<float>(u) - cam.cx) * depth_mm / cam.fx,
                            (static_cast<float>(v) - cam.cy) * depth_mm / cam.fy,
                            depth_mm,
                            1.f;
            const Eigen::Vector3f world_point = (camera_to_world * camera_point).head<3>();
            for (size_t d = 0; d < 3; ++d) {
                bbox_min[d] = std::min(bbox_min[d], world_point[d]);
                bbox_max[d] = std::max(bbox_max[d], world_point[d]);
            }
            sum += world_point;

            Eigen::Vector3f normal = Eigen::Vector3f::Zero();
            normal[2] = 1.0f;
            const auto *row = rgb.ptr<cv::Vec3b>(v);
            std::array<unsigned char, 3> color = {{
                    row[u][2],
                    row[u][1],
                    row[u][0]
            }};
            auto surfel = std::make_shared<inseg_lib::Surfel>(world_point,
                                                              normal,
                                                              5.0f,
                                                              1.0f,
                                                              0);
            surfel->color = color;
            surfel->is_valid = true;
            surfel->is_stable = true;
            surfels.emplace_back(std::move(surfel));
        }
    }

    const size_t min_points = static_cast<size_t>(std::max(1, config ? config->tele_yolo_object_min_points : 50));
    if (surfels.size() < min_points) return false;

    observation.class_idx = det.class_idx;
    observation.score = det.score;
    observation.centroid = sum / static_cast<float>(surfels.size());
    observation.bbox_min = bbox_min;
    observation.bbox_max = bbox_max;
    observation.surfels = std::move(surfels);
    return true;
}

static bool BuildTeleYoloObjectObservationFromRawMap(
        const TeleYoloDetection &det,
        const cv::Mat &depth,
        const Eigen::Matrix4f &world_to_camera,
        const CameraParameters &cam,
        const std::vector<SurfelPtr> &raw_surfels,
        const ConfigPSLAM *config,
        TeleYoloObjectObservation &observation) {
    observation = TeleYoloObjectObservation{};
    if (det.mask.empty() || depth.empty() || raw_surfels.empty()) return false;
    if (cam.fx <= 1e-6f || cam.fy <= 1e-6f) return false;

    const int max_points = std::max(1, config ? config->tele_yolo_object_max_points_per_mask : 3000);
    const float depth_tolerance_mm = 80.0f;
    Eigen::Vector3f sum = Eigen::Vector3f::Zero();
    Eigen::Vector3f bbox_min, bbox_max;
    bbox_min.setConstant(std::numeric_limits<float>::max());
    bbox_max.setConstant(std::numeric_limits<float>::lowest());

    std::vector<SurfelPtr> surfels;
    surfels.reserve(static_cast<size_t>(std::min<int>(max_points, raw_surfels.size())));
    const cv::Rect image_rect(0, 0, depth.cols, depth.rows);
    const cv::Rect det_box = det.box & image_rect;
    if (det_box.area() <= 0) return false;

    for (const auto &surfel : raw_surfels) {
        if (!surfel || !surfel->is_valid || !surfel->is_stable) continue;
        Eigen::Vector4f point;
        point << surfel->pos[0], surfel->pos[1], surfel->pos[2], 1.f;
        const Eigen::Vector4f camera_point = world_to_camera * point;
        const float z = camera_point[2];
        if (z <= 1e-4f) continue;
        const int u = static_cast<int>(std::round(cam.fx * (camera_point[0] / z) + cam.cx));
        const int v = static_cast<int>(std::round(cam.fy * (camera_point[1] / z) + cam.cy));
        if (!det_box.contains(cv::Point(u, v))) continue;
        if (det.mask.at<unsigned char>(v, u) == 0) continue;

        float depth_mm = 0.f;
        if (!ReadDepthMillimeters(depth, v, u, depth_mm)) continue;
        if (std::abs(depth_mm - z) > depth_tolerance_mm) continue;

        for (size_t d = 0; d < 3; ++d) {
            bbox_min[d] = std::min(bbox_min[d], surfel->pos[d]);
            bbox_max[d] = std::max(bbox_max[d], surfel->pos[d]);
        }
        sum += surfel->pos;
        surfels.emplace_back(surfel);
        if (static_cast<int>(surfels.size()) >= max_points) break;
    }

    const size_t min_points = static_cast<size_t>(std::max(1, config ? config->tele_yolo_object_min_points : 50));
    if (surfels.size() < min_points) return false;

    observation.class_idx = det.class_idx;
    observation.score = det.score;
    observation.centroid = sum / static_cast<float>(surfels.size());
    observation.bbox_min = bbox_min;
    observation.bbox_max = bbox_max;
    observation.surfels = std::move(surfels);
    return true;
}

// Efficient multi-detection observation builder using InSeg's per-pixel label map.
// For each detection: scans YOLO mask pixels → collects matching InSeg segment labels →
// then in a SINGLE pass over raw_surfels routes each surfel to matching detections.
// Returns one observation per detection (surfels empty if no match).
// Unlike BuildTeleYoloObjectObservationFromRawMap this gives ALL accumulated InSeg
// surfels for the matched segments (not just currently-visible ones), matching
// InSeg's dense multi-frame reconstruction quality.
// Backproject depth pixels inside each YOLO mask into 3-D world space.
// Points are subsampled with a stride so each detection stays within max_points.
static void BuildTeleYoloObjectObservationsFromMaskDepth(
        const std::vector<TeleYoloDetection> &detections,
        const cv::Mat &depth,
        const cv::Mat &rgb,                   // RGB (not BGR) — already converted in SetVisualFrame
        const Eigen::Matrix4f &world_to_camera,
        const CameraParameters &cam,
        const ConfigPSLAM *config,
        std::vector<TeleYoloObjectObservation> &observations)
{
    const int max_points = std::max(1, config ? config->tele_yolo_object_max_points_per_mask : 3000);
    const size_t min_points = static_cast<size_t>(std::max(1, config ? config->tele_yolo_object_min_points : 50));
    const size_t n_det = detections.size();
    observations.assign(n_det, TeleYoloObjectObservation{});
    if (n_det == 0 || depth.empty()) return;
    if (cam.fx <= 1e-6f || cam.fy <= 1e-6f) return;

    const Eigen::Matrix4f camera_to_world = world_to_camera.inverse();
    const cv::Rect img_rect(0, 0, depth.cols, depth.rows);
    const bool has_rgb = !rgb.empty() && rgb.rows == depth.rows && rgb.cols == depth.cols;

    for (size_t di = 0; di < n_det; ++di) {
        const auto &det = detections[di];
        if (det.mask.empty()) continue;
        const cv::Rect box = det.box & img_rect;
        if (box.area() <= 0) continue;

        // Choose stride so we sample at most max_points pixels from the bounding box.
        const int stride = std::max(1, static_cast<int>(
            std::ceil(std::sqrt(static_cast<float>(box.width * box.height) /
                                static_cast<float>(max_points)))));

        auto &obs = observations[di];
        obs.class_idx = det.class_idx;
        obs.score     = det.score;

        // Pass 1: collect valid mask depths to compute median (depth consistency filter).
        std::vector<float> mask_depths;
        mask_depths.reserve(static_cast<size_t>(
            (box.width / stride + 1) * (box.height / stride + 1)));
        for (int v = box.y; v < box.y + box.height; v += stride)
            for (int u = box.x; u < box.x + box.width; u += stride) {
                if (det.mask.at<uint8_t>(v, u) == 0) continue;
                float d_mm = 0.f;
                if (!ReadDepthMillimeters(depth, v, u, d_mm) || d_mm <= 0.f) continue;
                mask_depths.push_back(d_mm);
            }
        if (mask_depths.empty()) continue;
        const size_t mid_idx = mask_depths.size() / 2;
        std::nth_element(mask_depths.begin(), mask_depths.begin() + mid_idx, mask_depths.end());
        const float median_depth_mm = mask_depths[mid_idx];
        constexpr float kDepthTolMm = 150.0f;

        // Pass 2: backproject only pixels within kDepthTolMm of the median.
        Eigen::Vector3f sum = Eigen::Vector3f::Zero();
        Eigen::Vector3f bmin = Eigen::Vector3f::Constant(std::numeric_limits<float>::max());
        Eigen::Vector3f bmax = Eigen::Vector3f::Constant(std::numeric_limits<float>::lowest());
        bool capped = false;
        for (int v = box.y; v < box.y + box.height && !capped; v += stride) {
            for (int u = box.x; u < box.x + box.width && !capped; u += stride) {
                if (det.mask.at<uint8_t>(v, u) == 0) continue;
                float d_mm = 0.f;
                if (!ReadDepthMillimeters(depth, v, u, d_mm) || d_mm <= 0.f) continue;
                if (std::abs(d_mm - median_depth_mm) > kDepthTolMm) continue;

                const float z = d_mm;
                const float x = (static_cast<float>(u) - cam.cx) * z / cam.fx;
                const float y = (static_cast<float>(v) - cam.cy) * z / cam.fy;
                const Eigen::Vector3f world_pos =
                    (camera_to_world * Eigen::Vector4f(x, y, z, 1.f)).head<3>();

                auto s = std::make_shared<inseg_lib::Surfel>();
                s->pos   = world_pos;
                s->label = 0;
                s->is_valid = true;
                s->is_stable = false;
                s->index_in_graph = -1;
                if (has_rgb) {
                    const auto &px = rgb.at<cv::Vec3b>(v, u);
                    s->color = {{px[0], px[1], px[2]}};
                }

                obs.surfels.push_back(std::move(s));
                sum += world_pos;
                for (int d = 0; d < 3; ++d) {
                    bmin[d] = std::min(bmin[d], world_pos[d]);
                    bmax[d] = std::max(bmax[d], world_pos[d]);
                }
                if (static_cast<int>(obs.surfels.size()) >= max_points) capped = true;
            }
        }
        if (obs.surfels.size() < min_points) { obs.surfels.clear(); continue; }
        obs.centroid = sum / static_cast<float>(obs.surfels.size());
        obs.bbox_min = bmin;
        obs.bbox_max = bmax;
    }
}

static uint64_t TeleRawSurfelKey(const SurfelPtr &surfel, float voxel_size_mm) {
    // Voxel-grid key: pack (x,y,z) voxel indices into a uint64_t (20 bits each).
    // 20 bits covers ±524288 voxels; at 30mm that is ±15 km — more than sufficient.
    const uint32_t x = static_cast<uint32_t>(static_cast<int32_t>(std::floor(surfel->pos[0] / voxel_size_mm))) & 0xFFFFF;
    const uint32_t y = static_cast<uint32_t>(static_cast<int32_t>(std::floor(surfel->pos[1] / voxel_size_mm))) & 0xFFFFF;
    const uint32_t z = static_cast<uint32_t>(static_cast<int32_t>(std::floor(surfel->pos[2] / voxel_size_mm))) & 0xFFFFF;
    return (static_cast<uint64_t>(x) << 40) | (static_cast<uint64_t>(y) << 20) | static_cast<uint64_t>(z);
}

GraphPredictor::GraphPredictor(const ConfigPSLAM *configPslam):mConfigPslam(configPslam), mUpdated(false){
    source_to_target = false;
    auto path_onnx = ResolveTeleOnnxModuleDir(configPslam->pth_model);
    mbThread = configPslam->use_thread;
//    mConfigPaths = std::make_unique<ONNX_Model_PATHS>(path_onnx);
    mMemoryInfo = std::make_unique<Ort::MemoryInfo>(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));
//    mPointNetGCN = std::make_unique<PointNetGCN>(mConfigPaths.get(),mMemoryInfo.get(),mbVerbose);

    if (IsTeleOnnxModuleDir(path_onnx)) {
        mUseTeleModules = true;
        mTeleModulePath = path_onnx;
        mTeleHasForegroundFilter = HasTeleModule(path_onnx, "foreground_filter");
        mTeleHasObjPredictor = HasTeleModule(path_onnx, "obj_predictor");
        mTeleRelationOnly = !mTeleHasForegroundFilter || !mTeleHasObjPredictor;
        mUseTeleYolo = mTeleRelationOnly &&
                       mConfigPslam &&
                       !mConfigPslam->tele_yolo_model_path.empty() &&
                       FileExists(mConfigPslam->tele_yolo_model_path);
        InitTeleModules(path_onnx);
        LoadTeleLabels(path_onnx);
        mTeleParams["label_type"] = json11::Json("Tele");
        mTeleParams["n_layers"] = json11::Json(2);
        mTeleParams["USE_GCN"] = json11::Json(true);
        mTeleParams["USE_RGB"] = json11::Json(false);
        mTeleParams["USE_NORMAL"] = json11::Json(false);
        mTeleParams["dim_edge"] = json11::Json(11);
        mTeleParams["dim_o_f"] = json11::Json(256);
        mTeleParams["dim_r_f"] = json11::Json(256);
        mTeleParams["multi_rel_outputs"] = json11::Json(true);
        if (mTeleRelationOnly)
            mTeleParams["node_label_source"] = json11::Json("yolo_proposal");
        SCLOG(INFO) << "Loaded Tele-SGFN ONNX modules from " << path_onnx;
        if (mTeleRelationOnly) {
            SCLOG(INFO) << "Tele-SGFN relation-only mode enabled; "
                        << "foreground filter and object predictor are bypassed.";
            if (mUseTeleYolo)
                SCLOG(INFO) << "Tele YOLO proposals loaded from "
                            << mConfigPslam->tele_yolo_model_path;
            else
                SCLOG(WARNING) << "Tele relation-only mode has no usable YOLO ONNX model; "
                               << "no new node labels will be assigned.";
        }
        return;
    }

    mEatGCN = std::make_unique<EatGCN>(path_onnx, mMemoryInfo.get());

    /// Read labels
    {
        std::fstream file(path_onnx+"classes.txt", std::ios::in);
        if(!file.is_open())SCLOG(ERROR) << "Cannot open class file at path " << path_onnx+"classes.txt";
        std::string line;
        std::vector<std::string> labels;
        while (std::getline(file, line)) labels.push_back(line);

        SCLOG(VERBOSE) << "Labels";
        for(size_t i=0;i<labels.size();++i) {
            mLabels.insert({i, labels[i]});
            mLabelsName2Idx.insert({labels[i],i});
            SCLOG(VERBOSE) << i << ": " << labels[i];
        }
    }
    {
        std::fstream file(path_onnx+"relationships.txt", std::ios::in);
        if(!file.is_open())SCLOG(ERROR) << "Cannot open relationship file at path " << path_onnx+"relationships.txt";
        std::string line;
        std::vector<std::string> labels;
        while (std::getline(file, line)) labels.push_back(line);
        SCLOG(VERBOSE) << "Relationships";
        for(size_t i=0;i<labels.size();++i) {
            mRelationships.insert({i,labels[i]});
            mRelationshipsName2Idx.insert({labels[i],i});
            SCLOG(VERBOSE) << i << ": " << labels[i];
        }
    }
}

void GraphPredictor::InitTeleModules(const std::string &path_onnx) {
    mTeleOrtEnv = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "tele_sgfn");
    auto make_session_options = []() {
        Ort::SessionOptions sessionOptions;
        sessionOptions.SetIntraOpNumThreads(1);
        sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        sessionOptions.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        return sessionOptions;
    };

    // Try to load GNN modules on CUDA; fall back to CPU if unavailable.
    bool loaded_gnn_on_cuda = false;
    auto gnnSessionOptions = make_session_options();
    try {
        OrtCUDAProviderOptions cudaOptions{};
        cudaOptions.device_id = 0;
        gnnSessionOptions.AppendExecutionProvider_CUDA(cudaOptions);
        loaded_gnn_on_cuda = true;
    } catch (const std::exception &e) {
        SCLOG(WARNING) << "Failed to initialize Tele GNN modules on CUDA; falling back to CPU. "
                       << e.what();
        gnnSessionOptions = make_session_options();
    }

    std::vector<std::string> module_names = {
            "obj_encoder",
            "node_geometry",
            "node_feature_fuser",
            "edge_encoder",
            "gnn",
            "rel_predictor"
    };
    if (mTeleHasForegroundFilter) module_names.emplace_back("foreground_filter");
    if (mTeleHasObjPredictor) module_names.emplace_back("obj_predictor");
    for (const auto &name : module_names) {
        const auto path = path_onnx + name + ".onnx";
        mTeleSessions[name] = std::make_unique<Ort::Session>(*mTeleOrtEnv, path.c_str(), gnnSessionOptions);
    }
    SCLOG(INFO) << "Loaded Tele-SGFN GNN modules with "
                << (loaded_gnn_on_cuda ? "CUDA" : "CPU") << " execution provider.";

    if (mUseTeleYolo) {
        bool loaded_yolo_on_cuda = false;
        try {
            auto yoloSessionOptions = make_session_options();
            OrtCUDAProviderOptions cudaOptions{};
            cudaOptions.device_id = 0;
            yoloSessionOptions.AppendExecutionProvider_CUDA(cudaOptions);
            mTeleSessions["yolo"] =
                    std::make_unique<Ort::Session>(*mTeleOrtEnv,
                                                   mConfigPslam->tele_yolo_model_path.c_str(),
                                                   yoloSessionOptions);
            loaded_yolo_on_cuda = true;
        } catch (const std::exception &e) {
            SCLOG(WARNING) << "Failed to initialize Tele YOLO on CUDA; falling back to CPU. "
                           << e.what();
            auto yoloSessionOptions = make_session_options();
            mTeleSessions["yolo"] =
                    std::make_unique<Ort::Session>(*mTeleOrtEnv,
                                                   mConfigPslam->tele_yolo_model_path.c_str(),
                                                   yoloSessionOptions);
        }
        SCLOG(INFO) << "Loaded Tele YOLO segmentation ONNX from "
                    << mConfigPslam->tele_yolo_model_path
                    << " with " << (loaded_yolo_on_cuda ? "CUDA" : "CPU")
                    << " execution provider.";
    }

    // Start async YOLO thread if YOLO session is available.
    if (mUseTeleYolo && mTeleSessions.count("yolo")) {
        mYoloThreadStop = false;
        mYoloThread = std::async(std::launch::async, &GraphPredictor::YoloThread_IMPL, this);
        SCLOG(INFO) << "YOLO async thread started.";
    }

    if (mTeleRelationOnly) {
        mUseTeleVisualEncoder = false;
        return;
    }

    const auto visual_path = FindTeleVisualEncoderPath(path_onnx);
    if (!visual_path.empty()) {
        bool loaded_visual_on_cuda = false;
        try {
            auto visualSessionOptions = make_session_options();
            OrtCUDAProviderOptions cudaOptions{};
            cudaOptions.device_id = 0;
            visualSessionOptions.AppendExecutionProvider_CUDA(cudaOptions);
            mTeleSessions["visual_encoder"] =
                    std::make_unique<Ort::Session>(*mTeleOrtEnv, visual_path.c_str(), visualSessionOptions);
            loaded_visual_on_cuda = true;
        } catch (const std::exception &e) {
            SCLOG(WARNING) << "Failed to initialize Tele visual encoder on CUDA; falling back to CPU. "
                           << e.what();
            auto visualSessionOptions = make_session_options();
            mTeleSessions["visual_encoder"] =
                    std::make_unique<Ort::Session>(*mTeleOrtEnv, visual_path.c_str(), visualSessionOptions);
        }
        mUseTeleVisualEncoder = true;
        SCLOG(INFO) << "Loaded Tele visual encoder ONNX from " << visual_path
                    << " with " << (loaded_visual_on_cuda ? "CUDA" : "CPU") << " execution provider.";
    } else {
        SCLOG(WARNING) << "Tele visual encoder ONNX not found; visual features will be zeros.";
    }
}

void GraphPredictor::YoloThread_IMPL() {
    try {
    while (!mYoloThreadStop) {
        YoloAsyncInput input;
        {
            std::unique_lock<std::mutex> lock(mMutYoloInput);
            mCvYoloInput.wait(lock, [this]{ return mYoloInputReady || mYoloThreadStop.load(); });
            if (mYoloThreadStop) break;
            input = std::move(mYoloInput);
            mYoloInputReady = false;
        }
        // Run YOLO on the submitted frame.
        const auto _yolo_t0 = std::chrono::steady_clock::now();
        const auto detections = RunTeleYoloSegmentation(
                *mTeleSessions.at("yolo"),
                mMemoryInfo.get(),
                input.rgb,
                static_cast<int>(mLabels.size()),
                mConfigPslam->tele_yolo_conf_threshold,
                mConfigPslam->tele_yolo_iou_threshold,
                mConfigPslam->tele_yolo_min_mask_area);

        // Build visualization.
        cv::Mat viz;
        cv::cvtColor(input.rgb, viz, cv::COLOR_RGB2BGR);
        for (const auto &det : detections) {
            if (det.mask.empty()) continue;
            const cv::Vec3b color = YoloClassColor(det.class_idx);
            cv::Mat overlay = viz.clone();
            overlay.setTo(cv::Scalar(color[0], color[1], color[2]), det.mask);
            cv::addWeighted(viz, 0.55, overlay, 0.45, 0.0, viz);
            cv::rectangle(viz, det.box, cv::Scalar(color[0], color[1], color[2]), 2);
            const std::string label_text =
                (mLabels.count(static_cast<size_t>(det.class_idx))
                     ? mLabels.at(static_cast<size_t>(det.class_idx))
                     : std::to_string(det.class_idx))
                + " " + std::to_string(int(det.score * 100 + 0.5f)) + "%";
            int baseline = 0;
            const auto text_size = cv::getTextSize(label_text, cv::FONT_HERSHEY_SIMPLEX, 0.45f, 1, &baseline);
            const int ty = std::max(det.box.y - 4, text_size.height + 2);
            cv::rectangle(viz,
                          cv::Point(det.box.x, ty - text_size.height - 2),
                          cv::Point(det.box.x + text_size.width, ty + baseline),
                          cv::Scalar(color[0], color[1], color[2]), cv::FILLED);
            cv::putText(viz, label_text, cv::Point(det.box.x, ty),
                        cv::FONT_HERSHEY_SIMPLEX, 0.45f, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
        }

        // Store paired result (detections + matching rgb/depth/pose from same frame).
        {
            std::unique_lock<std::mutex> lock(mMutYoloResult);
            mYoloResult.detections      = detections;
            mYoloResult.rgb             = std::move(input.rgb);
            mYoloResult.depth           = std::move(input.depth);
            mYoloResult.world_to_camera = input.world_to_camera;
            mYoloResult.cam             = input.cam;
            mYoloResult.frame_idx       = input.frame_idx;
            mYoloResult.visualization   = std::move(viz);
            mYoloResultReady            = true;
        }
        mCvYoloResult.notify_one();
        mYoloInferTotalMs += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - _yolo_t0).count();
        ++mYoloInferCount;
    }
    } catch (const std::exception &e) {
        SCLOG(ERROR) << "[YoloThread] fatal exception: " << e.what();
    } catch (...) {
        SCLOG(ERROR) << "[YoloThread] unknown fatal exception.";
    }
}

void GraphPredictor::LoadTeleLabels(const std::string &path_onnx) {
    const auto model_path = WithTrailingSlash(path_onnx);
    const auto parent_path = ParentDir(model_path);
    const auto dataset_path = WithTrailingSlash(mConfigPslam->pth_dataset);

    const auto class_labels = ReadNonEmptyLines({
            model_path + "classes.txt",
            parent_path + "classes.txt",
            dataset_path + "classes.txt",
            "/media/gao/gao_ssd/teledataset_/classes.txt"
    }, "Tele class labels");
    for (size_t i = 0; i < class_labels.size(); ++i) {
        mLabels.insert({i, class_labels[i]});
        mLabelsName2Idx.insert({class_labels[i], i});
    }

    auto relation_labels = ReadNonEmptyLines({
            model_path + "relationships.txt",
            parent_path + "relationships.txt",
            dataset_path + "relationships.txt",
            "/media/gao/gao_ssd/teledataset_/relationships.txt"
    }, "Tele relationship labels");
    relation_labels.erase(std::remove(relation_labels.begin(), relation_labels.end(), Edge::None()),
                          relation_labels.end());
    if (std::find(relation_labels.begin(), relation_labels.end(), Edge::Same()) == relation_labels.end())
        relation_labels.emplace_back(Edge::Same());
    for (size_t i = 0; i < relation_labels.size(); ++i) {
        mRelationships.insert({i, relation_labels[i]});
        mRelationshipsName2Idx.insert({relation_labels[i], i});
    }
}

std::vector<Ort::Value> GraphPredictor::RunTeleModule(const std::string &name,
                                                      std::vector<Ort::Value> &inputs) const {
    std::vector<const char *> input_names;
    std::vector<const char *> output_names;
    if (name == "obj_encoder") {
        input_names = {"node_pts"};
        output_names = {"node_point_feature"};
    } else if (name == "node_geometry") {
        input_names = {"node_descriptor", "position_geo"};
        output_names = {"node_geo_feature"};
    } else if (name == "node_feature_fuser") {
        input_names = {"node_fuser_input"};
        output_names = {"fused_node_feature"};
    } else if (name == "foreground_filter") {
        input_names = {"node_point_feature", "foreground_geo", "visual_feature"};
        output_names = {"foreground_logits"};
    } else if (name == "edge_encoder") {
        input_names = {"node_descriptor", "edge_index"};
        output_names = {"edge_feature"};
    } else if (name == "gnn") {
        input_names = {"node_feature", "edge_feature", "edge_index"};
        output_names = {"gnn_node_feature", "gnn_edge_feature"};
    } else if (name == "obj_predictor") {
        input_names = {"node_feature"};
        output_names = {"node_logits"};
    } else if (name == "rel_predictor") {
        input_names = {"edge_feature"};
        output_names = {"edge_logits"};
    } else if (name == "visual_encoder") {
        input_names = {"rgb_crop"};
        output_names = {"visual_feature"};
    } else {
        throw std::runtime_error("Unknown Tele ONNX module: " + name);
    }

    return mTeleSessions.at(name)->Run(Ort::RunOptions{nullptr},
                                       input_names.data(), inputs.data(), input_names.size(),
                                       output_names.data(), output_names.size());
}

GraphPredictor::~GraphPredictor(){
    // Stop YOLO async thread.
    mYoloThreadStop = true;
    mCvYoloInput.notify_all();
    if (mYoloThread.valid())
        mYoloThread.get();

    mbShouldStop=true;
    if(mbThread) {
        if(mThread.valid()) {
            while (mThread.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready)
                mThread.wait_for(std::chrono::milliseconds(50));
            mThread.get();
        }
    }
}

std::shared_ptr<PSLAM::GraphPredictor::Prediction> GraphPredictor::Predict(
        const std::shared_ptr<GCNINPUTDATA>& data) {
    if (mUseTeleModules) {
        throw std::runtime_error("GraphPredictor::Predict is not used by the Tele-SGFN module pipeline.");
    }
    {
        std::tuple<MemoryBlock2D, MemoryBlock2D> result;
        auto& edge_index = std::get<2>(*data);
        auto& idx2seg = std::get<3>(*data);
        result = mEatGCN->Run(std::get<0>(*data), std::get<1>(*data),std::get<2>(*data));
        auto& cls_obj = std::get<0>(result);
        auto& cls_rel = std::get<1>(result);


        std::map<size_t,std::map<std::string,float>> output_objs;
        for(size_t i=0;i<cls_obj.mDims.x;++i){
            std::map<std::string,float> m;
            for(size_t j=0;j<cls_obj.mDims.y;++j){
                m[mLabels.at(j)] = cls_obj.at<float>(i,j);
            }
            output_objs[idx2seg.at(i)] = std::move(m);
        }
//        std::map<size_t,size_t> output_objs;
//        for(size_t i=0;i<cls_obj.size();++i){
//            output_objs[idx2seg.at(i)] = cls_obj.at(i)+1;
//        }

        std::vector<EdgePtr> output_edges;
        for(size_t i=0;i<cls_rel.mDims.x;++i){
            output_edges.emplace_back(new Edge());
            auto &edge = output_edges.back();
            edge->nodeFrom = idx2seg.at( edge_index.at<int64_t>(0,i) );
            edge->nodeTo = idx2seg.at( edge_index.at<int64_t>(1,i) );
            for(size_t j=0;j<cls_rel.mDims.y;++j) {
                edge->labelProp[mRelationships.at(j)] = cls_rel.at<float>(i,j);
            }
        }

        return std::make_shared<Prediction>(
                std::move(output_objs), std::move(output_edges), std::move(std::get<std::map<int, std::pair<size_t,size_t>>>(*data)));
    }
}

void GraphPredictor::RUN(Graph *graph) {
    mpGraph = graph;
    mbShouldStop = false;
    if(mbThread) {
        if (mThread.valid()) {
            while (mThread.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready)
                mThread.wait_for(std::chrono::milliseconds(50));
            mThread.get();
        } else {
            mThread = std::async(std::launch::async,
                                 std::bind(&PSLAM::GraphPredictor::Process_IMPL, this));
        }

    } else {
        Process_IMPL();
    }
}

void GraphPredictor::SetVisualFrame(const cv::Mat &rgb,
                                    const cv::Mat &depth,
                                    const Eigen::Matrix4f &pose,
                                    const CameraParameters &cam,
                                    const Eigen::Matrix4f *tele_network_pose,
                                    int frame_idx) {
    if (!mUseTeleModules) return;
    std::unique_lock<std::mutex> lock(mMutVisualFrame);
    if (rgb.empty()) {
        mHasVisualFrame = false;
        mHasTeleNetworkPose = false;
        mVisualRGB.release();
        mVisualDepth.release();
        return;
    }

    if (rgb.channels() == 3)
        cv::cvtColor(rgb, mVisualRGB, cv::COLOR_BGR2RGB);
    else if (rgb.channels() == 4)
        cv::cvtColor(rgb, mVisualRGB, cv::COLOR_BGRA2RGB);
    else
        cv::cvtColor(rgb, mVisualRGB, cv::COLOR_GRAY2RGB);
    if (depth.empty()) {
        mVisualDepth.release();
    } else if (depth.channels() == 1) {
        mVisualDepth = depth.clone();
    } else {
        cv::cvtColor(depth, mVisualDepth, cv::COLOR_BGR2GRAY);
    }
    mVisualPose = pose;
    mVisualFrameIndex = frame_idx;
    if (tele_network_pose) {
        mTeleNetworkPose = *tele_network_pose;
        mHasTeleNetworkPose = true;
    } else {
        mHasTeleNetworkPose = false;
    }
    mVisualCam = cam;
    mHasVisualFrame = true;
}

bool GraphPredictor::UpdateTeleYoloObjectGraph(Graph *object_graph, int frame_idx) {
    if (!object_graph || !mUseTeleModules || !mTeleRelationOnly ||
        !mUseTeleYolo || mTeleSessions.find("yolo") == mTeleSessions.end())
        return false;

    // --- Submit current frame to YOLO async thread (non-blocking) ---
    // Copy frame data under mMutVisualFrame first, then submit under mMutYoloInput
    // (never hold both locks at once to avoid potential deadlock).
    {
        cv::Mat sub_rgb, sub_depth;
        Eigen::Matrix4f sub_pose;
        CameraParameters sub_cam{};
        {
            std::unique_lock<std::mutex> lock(mMutVisualFrame);
            if (!mHasVisualFrame || mVisualRGB.empty() || mVisualDepth.empty()) return false;
            sub_rgb   = mVisualRGB.clone();
            sub_depth = mVisualDepth.clone();
            sub_pose  = mVisualPose;
            sub_cam   = mVisualCam;
        }
        {
            std::unique_lock<std::mutex> lock(mMutYoloInput);
            mYoloInput.rgb             = std::move(sub_rgb);
            mYoloInput.depth           = std::move(sub_depth);
            mYoloInput.world_to_camera = sub_pose;
            mYoloInput.cam             = sub_cam;
            mYoloInput.frame_idx       = frame_idx;
            mYoloInputReady            = true;
        }
        mCvYoloInput.notify_one();
    }

    // --- Consume YOLO result if ready (non-blocking) ---
    std::vector<TeleYoloDetection> detections;
    cv::Mat rgb, depth;
    Eigen::Matrix4f world_to_camera = Eigen::Matrix4f::Identity();
    CameraParameters cam{};
    {
        std::unique_lock<std::mutex> lock(mMutYoloResult);
        if (!mYoloResultReady) {
            if (!mFirstYoloReceived)
                mCvYoloResult.wait(lock, [this]{ return mYoloResultReady || mYoloThreadStop.load(); });
            else
                return false;
        }
        if (!mYoloResultReady) return false;
        mFirstYoloReceived = true;
        mLastYoloFrameIdx  = mYoloResult.frame_idx;
        detections         = std::move(mYoloResult.detections);
        rgb                = std::move(mYoloResult.rgb);
        depth              = std::move(mYoloResult.depth);
        world_to_camera    = mYoloResult.world_to_camera;
        cam                = mYoloResult.cam;
        mYoloVisualization = std::move(mYoloResult.visualization);
        mYoloResultReady   = false;
    }
    CTICK("[YoloObjectGraph]1.yolo_inference"); // timing slot kept for display (time is in thread)
    CTOCK("[YoloObjectGraph]1.yolo_inference");

    if (cam.width <= 0 || cam.height <= 0) return false;

    CTICK("[YoloObjectGraph]2.build_observations");
    // Backproject depth pixels inside each YOLO mask to get the object's 3-D point cloud.
    std::vector<TeleYoloObjectObservation> obs_by_det;
    BuildTeleYoloObjectObservationsFromMaskDepth(
            detections, depth, rgb, world_to_camera, cam,
            mConfigPslam, obs_by_det);
    // Per-observation SOR: filter current-frame points BEFORE accumulation.
    // This removes depth noise at mask boundaries without eroding the accumulated cloud.
    for (auto &obs : obs_by_det) {
        if (obs.surfels.size() < 10) continue;
        // Two passes at 1.5σ on the fresh observation points.
        for (int pass = 0; pass < 2; ++pass) {
            const size_t n = obs.surfels.size();
            if (n < 10) break;
            float sum_d = 0.f, sum_sq = 0.f;
            for (const auto &s : obs.surfels) sum_d += (s->pos - obs.centroid).norm();
            const float mean_d = sum_d / static_cast<float>(n);
            for (const auto &s : obs.surfels) {
                const float diff = (s->pos - obs.centroid).norm() - mean_d;
                sum_sq += diff * diff;
            }
            const float thr = mean_d + 1.5f * std::sqrt(sum_sq / static_cast<float>(n));
            const size_t before = obs.surfels.size();
            obs.surfels.erase(
                std::remove_if(obs.surfels.begin(), obs.surfels.end(),
                    [&](const SurfelPtr &s) {
                        return (s->pos - obs.centroid).norm() > thr;
                    }),
                obs.surfels.end());
            if (obs.surfels.size() == before) break;
            // Recompute centroid for next pass.
            Eigen::Vector3f new_cen = Eigen::Vector3f::Zero();
            for (const auto &s : obs.surfels) new_cen += s->pos;
            obs.centroid = new_cen / static_cast<float>(obs.surfels.size());
        }
    }

    // Compact: keep only valid observations.
    std::vector<TeleYoloObjectObservation> observations;
    observations.reserve(detections.size());
    for (auto &obs : obs_by_det)
        if (!obs.surfels.empty())
            observations.emplace_back(std::move(obs));
    CTOCK("[YoloObjectGraph]2.build_observations");


    CTICK("[YoloObjectGraph]3.track_matching_and_update");
    const float assoc_distance_mm =
            std::max(0.05f, mConfigPslam->tele_yolo_track_assoc_distance) * 1000.0f;
    const int max_age = mConfigPslam->tele_yolo_track_max_age;
    const int graph_id_offset = 100000;
    std::set<int> used_tracks;

    const int vis_max_points = std::max(1, mConfigPslam->tele_yolo_vis_max_points);

    // Helper: compute effective association distance between a track and an observation.
    // Returns 0 if the observation centroid is inside the track's accumulated bbox (with margin),
    // otherwise returns the centroid-to-centroid distance.
    auto track_obs_distance = [&](const TeleYoloObjectTrack &track,
                                  const TeleYoloObjectObservation &obs,
                                  float bbox_margin_mm) -> float {
        const float dist = (track.centroid - obs.centroid).norm();
        const bool inside = track.surfels.size() >= 10 &&
            obs.centroid[0] >= track.bbox_min[0] - bbox_margin_mm &&
            obs.centroid[0] <= track.bbox_max[0] + bbox_margin_mm &&
            obs.centroid[1] >= track.bbox_min[1] - bbox_margin_mm &&
            obs.centroid[1] <= track.bbox_max[1] + bbox_margin_mm &&
            obs.centroid[2] >= track.bbox_min[2] - bbox_margin_mm &&
            obs.centroid[2] <= track.bbox_max[2] + bbox_margin_mm;
        return inside ? 0.f : dist;
    };

    for (const auto &observation : observations) {
        int best_track_id = -1;
        float best_distance = std::numeric_limits<float>::max();

        // Pass 1: same-class association (standard distance threshold).
        for (const auto &track_pair : mTeleYoloTracks) {
            const auto &track = track_pair.second;
            if (track.class_idx != observation.class_idx) continue;
            if (used_tracks.count(track_pair.first)) continue;
            if (max_age >= 0 && track.last_seen_frame >= 0 &&
                frame_idx - track.last_seen_frame > max_age) continue;
            const float d = track_obs_distance(track, observation, assoc_distance_mm);
            if (d < best_distance) { best_distance = d; best_track_id = track_pair.first; }
        }

        // Pass 2: cross-class spatial overlap check.
        // If no same-class track was found, search ALL existing tracks (including those
        // already used by a same-class observation this frame).
        // • If the overlap is with an UNUSED track  → temporal class fusion: update its class.
        // • If the overlap is with a USED track     → duplicate detection of the same region;
        //   skip this observation entirely (don't create a new track).
        if (best_track_id < 0 || best_distance > assoc_distance_mm) {
            bool duplicate_of_used = false;
            for (const auto &track_pair : mTeleYoloTracks) {
                const auto &track = track_pair.second;
                if (track.class_idx == observation.class_idx) continue;
                if (max_age >= 0 && track.last_seen_frame >= 0 &&
                    frame_idx - track.last_seen_frame > max_age) continue;
                const float d = track_obs_distance(track, observation, 0.f);  // strict: no bbox margin
                if (d > assoc_distance_mm) continue;
                // Same spatial region already occupied by a different-class track → skip.
                duplicate_of_used = true;
                break;
            }
            if (duplicate_of_used) continue;  // skip observation — don't create duplicate track
        }

        if (best_track_id < 0 || best_distance > assoc_distance_mm) {
            best_track_id = mTeleYoloNextTrackId++;
            TeleYoloObjectTrack track;
            track.graph_id = graph_id_offset + best_track_id;
            track.class_idx = observation.class_idx;
            track.vis_surfels.reserve(static_cast<size_t>(vis_max_points));
            mTeleYoloTracks[best_track_id] = std::move(track);
        }
        used_tracks.insert(best_track_id);

        auto &track = mTeleYoloTracks.at(best_track_id);
        track.class_idx = observation.class_idx;
        track.score = observation.score;
        track.last_seen_frame = frame_idx;

        const int max_track_points = std::max(1, mConfigPslam->tele_yolo_track_max_points);
        const float voxel_mm = std::max(1.f, mConfigPslam->tele_yolo_object_voxel_size * 1000.f);

        // Per-observation SOR: remove background outliers before they enter the track.
        // Compute centroid of this observation's surfels, then discard points beyond 2σ.
        std::vector<SurfelPtr> obs_inliers;
        if (observation.surfels.size() >= 10) {
            Eigen::Vector3f obs_cen = Eigen::Vector3f::Zero();
            for (const auto &s : observation.surfels) obs_cen += s->pos;
            obs_cen /= static_cast<float>(observation.surfels.size());
            float obs_sum_d = 0.f, obs_sum_sq = 0.f;
            for (const auto &s : observation.surfels)
                obs_sum_d += (s->pos - obs_cen).norm();
            const float obs_mean_d = obs_sum_d / static_cast<float>(observation.surfels.size());
            for (const auto &s : observation.surfels) {
                const float diff = (s->pos - obs_cen).norm() - obs_mean_d;
                obs_sum_sq += diff * diff;
            }
            const float obs_thr = obs_mean_d +
                2.0f * std::sqrt(obs_sum_sq / static_cast<float>(observation.surfels.size()));
            for (const auto &s : observation.surfels)
                if ((s->pos - obs_cen).norm() <= obs_thr)
                    obs_inliers.push_back(s);
        } else {
            obs_inliers = observation.surfels;
        }

        // GNN buffer: coarse voxels from filtered observation surfels
        for (const auto &src : obs_inliers) {
            const uint64_t key = TeleRawSurfelKey(src, voxel_mm);
            auto iter = track.surfels.find(key);
            if (iter == track.surfels.end()) {
                if (static_cast<int>(track.surfels.size()) < max_track_points) {
                    auto dst = std::make_shared<inseg_lib::Surfel>(*src);
                    dst->SetLabel(track.graph_id);
                    dst->index_in_graph = -1;
                    dst->is_valid = true;
                    dst->is_stable = true;
                    track.surfels.emplace(key, std::move(dst));
                }
            } else {
                auto &dst = iter->second;
                dst->pos = src->pos;
                dst->normal = src->normal;
                dst->color = src->color;
                dst->confidence = std::max(dst->confidence, src->confidence);
                dst->is_valid = true;
                dst->is_stable = true;
            }
        }

        // Vis buffer: accumulate points across frames so the cloud grows as the
        // camera moves around the object (mask depth backprojection approach).
        for (const auto &src : observation.surfels) {
            if (static_cast<int>(track.vis_surfels.size()) >= vis_max_points) break;
            track.vis_surfels.push_back({src->pos, {{0, 0, 0}}, track.graph_id});
        }
    }

    for (auto iter = mTeleYoloTracks.begin(); iter != mTeleYoloTracks.end();) {
        if (max_age >= 0 &&
            iter->second.last_seen_frame >= 0 &&
            frame_idx - iter->second.last_seen_frame > max_age) {
            iter = mTeleYoloTracks.erase(iter);
        } else {
            ++iter;
        }
    }

    // Only recompute centroid/bbox for tracks updated this frame.
    for (int tid : used_tracks) {
        auto it = mTeleYoloTracks.find(tid);
        if (it == mTeleYoloTracks.end()) continue;
        auto &track = it->second;
        if (track.surfels.empty()) continue;
        Eigen::Vector3f sum = Eigen::Vector3f::Zero();
        Eigen::Vector3f bbox_min, bbox_max;
        bbox_min.setConstant(std::numeric_limits<float>::max());
        bbox_max.setConstant(std::numeric_limits<float>::lowest());
        for (const auto &surfel_pair : track.surfels) {
            const auto &surfel = surfel_pair.second;
            sum += surfel->pos;
            for (size_t d = 0; d < 3; ++d) {
                bbox_min[d] = std::min(bbox_min[d], surfel->pos[d]);
                bbox_max[d] = std::max(bbox_max[d], surfel->pos[d]);
            }
        }
        track.centroid = sum / static_cast<float>(track.surfels.size());
        track.bbox_min = bbox_min;
        track.bbox_max = bbox_max;
    }

    CTOCK("[YoloObjectGraph]3.track_matching_and_update");

    CTICK("[YoloObjectGraph]4.outlier_removal");
    // Safety-net SOR on the accumulated GNN buffer (single pass, 2.5σ).
    // Observation-level SOR already cleaned current-frame points before accumulation,
    // so this only catches rare stragglers from mis-associations across frames.
    for (int tid : used_tracks) {
        auto tit = mTeleYoloTracks.find(tid);
        if (tit == mTeleYoloTracks.end()) continue;
        auto &track = tit->second;
        const size_t n = track.surfels.size();
        if (n < 20) continue;

        Eigen::Vector3f cen = Eigen::Vector3f::Zero();
        for (const auto &sp : track.surfels) cen += sp.second->pos;
        cen /= static_cast<float>(n);

        float sum_d = 0.f, sum_sq = 0.f;
        for (const auto &sp : track.surfels)
            sum_d += (sp.second->pos - cen).norm();
        const float mean_d = sum_d / static_cast<float>(n);
        for (const auto &sp : track.surfels) {
            const float diff = (sp.second->pos - cen).norm() - mean_d;
            sum_sq += diff * diff;
        }
        const float thr = mean_d + 2.5f * std::sqrt(sum_sq / static_cast<float>(n));

        for (auto it2 = track.surfels.begin(); it2 != track.surfels.end(); ) {
            if ((it2->second->pos - cen).norm() > thr) {
                if (it2->second->index_in_graph >= 0)
                    object_graph->RemoveSurfelFromNode(it2->second);
                it2 = track.surfels.erase(it2);
            } else { ++it2; }
        }
        if (track.surfels.empty()) continue;

        // Recompute centroid/bbox from inliers.
        Eigen::Vector3f s = Eigen::Vector3f::Zero();
        Eigen::Vector3f bmin, bmax;
        bmin.setConstant(std::numeric_limits<float>::max());
        bmax.setConstant(std::numeric_limits<float>::lowest());
        for (const auto &sp : track.surfels) {
            s += sp.second->pos;
            for (int d = 0; d < 3; ++d) {
                bmin[d] = std::min(bmin[d], sp.second->pos[d]);
                bmax[d] = std::max(bmax[d], sp.second->pos[d]);
            }
        }
        track.centroid = s / static_cast<float>(track.surfels.size());
        track.bbox_min = bmin;
        track.bbox_max = bmax;

        // vis_surfels are already cleaned by per-observation SOR before accumulation;
        // no need to re-filter them here.
    }

    CTOCK("[YoloObjectGraph]4.outlier_removal");

    // Merge same-class tracks that overlap spatially.
    // After each frame's SOR+centroid update, any two same-class tracks whose centroids
    // are within assoc_distance of each other (or inside each other's bbox) are duplicates
    // of the same physical object seen from different viewpoints — absorb the younger into the older.
    {
        const int max_merge_pts = std::max(1, mConfigPslam->tele_yolo_track_max_points);
        for (auto it1 = mTeleYoloTracks.begin(); it1 != mTeleYoloTracks.end(); ++it1) {
            auto &t1 = it1->second;
            if (t1.surfels.size() < 10) continue;
            for (auto it2 = std::next(it1); it2 != mTeleYoloTracks.end(); ) {
                auto &t2 = it2->second;
                if (t2.class_idx != t1.class_idx || t2.surfels.size() < 10) { ++it2; continue; }
                const float dist = (t1.centroid - t2.centroid).norm();
                auto in_bbox = [&](const TeleYoloObjectTrack &ta, const Eigen::Vector3f &p) {
                    return p[0] >= ta.bbox_min[0] - assoc_distance_mm &&
                           p[0] <= ta.bbox_max[0] + assoc_distance_mm &&
                           p[1] >= ta.bbox_min[1] - assoc_distance_mm &&
                           p[1] <= ta.bbox_max[1] + assoc_distance_mm &&
                           p[2] >= ta.bbox_min[2] - assoc_distance_mm &&
                           p[2] <= ta.bbox_max[2] + assoc_distance_mm;
                };
                // Two same-class tracks covering different halves of a large object
                // (e.g. a sofa) may have centroids far apart and neither centroid
                // inside the other's bbox — catch them by checking 3D bbox overlap.
                auto bbox_iou = [&](const TeleYoloObjectTrack &ta,
                                    const TeleYoloObjectTrack &tb) -> float {
                    float inter_vol = 1.f;
                    for (int d = 0; d < 3; ++d) {
                        const float lo = std::max(ta.bbox_min[d], tb.bbox_min[d]);
                        const float hi = std::min(ta.bbox_max[d], tb.bbox_max[d]);
                        if (hi <= lo) return 0.f;
                        inter_vol *= (hi - lo);
                    }
                    float va = 1.f, vb = 1.f;
                    for (int d = 0; d < 3; ++d) {
                        va *= std::max(1e-3f, ta.bbox_max[d] - ta.bbox_min[d]);
                        vb *= std::max(1e-3f, tb.bbox_max[d] - tb.bbox_min[d]);
                    }
                    return inter_vol / (va + vb - inter_vol);
                };
                const std::string cls_name = (t1.class_idx >= 0 &&
                    mLabels.count(static_cast<size_t>(t1.class_idx)))
                    ? mLabels.at(static_cast<size_t>(t1.class_idx)) : "";
                const bool is_sofa = (cls_name == "sofa" ||
                                      cls_name == "cabinet" ||
                                      cls_name == "wall_cabinet" ||
                                      cls_name == "chest_of_drawers" ||
                                      cls_name == "fridge" ||
                                      cls_name == "tvstand");
                bool should_merge = false;
                if (cls_name == "kitchen_counter" || cls_name == "kitchen_cupboard") {
                    // Prior: only one instance per sequence — always merge duplicates.
                    should_merge = true;
                } else if (is_sofa) {
                    const float sofa_margin_mm = 1000.f;
                    bool bboxes_close = true;
                    for (int d = 0; d < 3; ++d) {
                        if (t1.bbox_max[d] + sofa_margin_mm < t2.bbox_min[d] ||
                            t2.bbox_max[d] + sofa_margin_mm < t1.bbox_min[d]) {
                            bboxes_close = false; break;
                        }
                    }
                    should_merge = bboxes_close;
                } else {
                    should_merge = (bbox_iou(t1, t2) >= 0.15f);
                }
                if (!should_merge) { ++it2; continue; }
                // Absorb t2 into t1 (keep t1 = older / lower track-id)
                for (auto &sp : t2.surfels) {
                    if (!t1.surfels.count(sp.first) && (int)t1.surfels.size() < max_merge_pts) {
                        sp.second->SetLabel(t1.graph_id);
                        sp.second->index_in_graph = -1;
                        t1.surfels.emplace(sp.first, sp.second);
                    }
                }
                for (const auto &vp : t2.vis_surfels)
                    if ((int)t1.vis_surfels.size() < vis_max_points)
                        t1.vis_surfels.push_back(vp);
                // Recompute t1 centroid/bbox after absorbing t2
                {
                    Eigen::Vector3f sm = Eigen::Vector3f::Zero();
                    t1.bbox_min.setConstant(std::numeric_limits<float>::max());
                    t1.bbox_max.setConstant(std::numeric_limits<float>::lowest());
                    for (const auto &sp : t1.surfels) {
                        sm += sp.second->pos;
                        for (int d = 0; d < 3; ++d) {
                            t1.bbox_min[d] = std::min(t1.bbox_min[d], sp.second->pos[d]);
                            t1.bbox_max[d] = std::max(t1.bbox_max[d], sp.second->pos[d]);
                        }
                    }
                    t1.centroid = sm / static_cast<float>(t1.surfels.size());
                }
                // Remove t2 node from graph only if it has already been registered.
                // Newly-created tracks in the current frame have a graph_id but their
                // graph node isn't created until step 5 (graph_rebuild), so guard here.
                {
                    bool t2_registered = false;
                    {
                        std::unique_lock<std::mutex> lock(object_graph->mMutNode);
                        t2_registered = object_graph->nodes.count(t2.graph_id) > 0;
                    }
                    if (t2_registered)
                        object_graph->RemoveNode(t2.graph_id);
                }
                it2 = mTeleYoloTracks.erase(it2);
            }
        }
    }

    CTICK("[YoloObjectGraph]5.graph_rebuild");
    size_t active_nodes = 0;
    const size_t min_points =
            static_cast<size_t>(std::max(1, mConfigPslam->tele_yolo_object_min_points));

    // Remove nodes for expired tracks
    {
        std::set<int> active_graph_ids;
        for (const auto &tp : mTeleYoloTracks)
            active_graph_ids.insert(tp.second.graph_id);
        std::vector<int> to_remove;
        {
            std::unique_lock<std::mutex> lock(object_graph->mMutNode);
            for (const auto &np : object_graph->nodes)
                if (active_graph_ids.find(np.first) == active_graph_ids.end())
                    to_remove.push_back(np.first);
        }
        for (int id : to_remove)
            object_graph->RemoveNode(id);
    }

    for (auto &track_pair : mTeleYoloTracks) {
        auto &track = track_pair.second;
        if (track.surfels.size() < min_points) continue;
        ++active_nodes;

        // Register any new surfels (handles first-appearance tracks; no-op for old tracks).
        for (const auto &surfel_pair : track.surfels) {
            const auto &surfel = surfel_pair.second;
            surfel->SetLabel(track.graph_id);
            if (surfel->index_in_graph < 0)
                surfel->index_in_graph = object_graph->Add(surfel);
        }

        // Only update YOLO prediction and mark node dirty for tracks seen this frame.
        if (used_tracks.find(track_pair.first) == used_tracks.end()) continue;

        NodePtr node;
        {
            std::unique_lock<std::mutex> lock(object_graph->mMutNode);
            auto iter = object_graph->nodes.find(track.graph_id);
            if (iter == object_graph->nodes.end()) continue;
            node = iter->second;
        }

        node->centroid = track.centroid;
        node->bbox_min = track.bbox_min;
        node->bbox_max = track.bbox_max;
        node->bNeedCheckNeighbor = true;
        {
            std::unique_lock<std::mutex> lock(object_graph->mMutThis);
            object_graph->nodes_to_update.insert(track.graph_id);
        }

        node->instance_idx = track.graph_id;
        if (track.class_idx >= 0 && static_cast<size_t>(track.class_idx) < mLabels.size()) {
            std::map<std::string, float> cls_prob;
            std::map<std::string, std::pair<size_t, size_t>> size_edge;
            const auto fallback = std::make_pair(node->surfels.size(), node->neighbors.size());
            for (const auto &label_pair : mLabels) {
                cls_prob[label_pair.second] = 0.f;
                size_edge[label_pair.second] = fallback;
            }
            cls_prob[mLabels.at(static_cast<size_t>(track.class_idx))] = track.score;
            node->UpdatePrediction(cls_prob, size_edge, false, 0.0f);
        }
    }

    CTOCK("[YoloObjectGraph]5.graph_rebuild");

    // Rebuild snapshot only when total vis count has actually changed.
    {
        size_t cur_total = 0;
        for (const auto &tp : mTeleYoloTracks)
            cur_total += tp.second.vis_surfels.size();

        if (cur_total != mTeleYoloTotalVisCount) {
            VisPointVec vis_tmp;
            vis_tmp.reserve(cur_total);
            for (const auto &tp : mTeleYoloTracks)
                vis_tmp.insert(vis_tmp.end(),
                               tp.second.vis_surfels.begin(),
                               tp.second.vis_surfels.end());
            std::unique_lock<std::mutex> lock(mMutVisSurfels);
            mTeleYoloVisSurfels     = std::move(vis_tmp);
            mTeleYoloTotalVisCount  = cur_total;
        }
    }

    SCLOG(VERBOSE) << "Tele YOLO object graph: " << detections.size()
                   << " masks, " << observations.size()
                   << " raw-map objects, " << active_nodes
                   << " active tracks.";
    return active_nodes > 0;
}

bool GraphPredictor::ComputeTeleFrameSceneBounds(
        Eigen::Vector3f &scene_min,
        Eigen::Vector3f &scene_max,
        const Eigen::Matrix3f &geometry_rotation,
        const Eigen::Vector3f &geometry_translation,
        bool use_geometry_transform) {
    cv::Mat depth;
    Eigen::Matrix4f world_to_camera;
    CameraParameters cam{};
    {
        std::unique_lock<std::mutex> lock(mMutVisualFrame);
        if (!mHasVisualFrame || mVisualDepth.empty()) return false;
        depth = mVisualDepth.clone();
        world_to_camera = mVisualPose;
        cam = mVisualCam;
    }

    if (cam.fx <= 1e-6f || cam.fy <= 1e-6f || depth.empty()) return false;
    const Eigen::Matrix4f camera_to_world = world_to_camera.inverse();
    scene_min.setConstant(std::numeric_limits<float>::max());
    scene_max.setConstant(std::numeric_limits<float>::lowest());

    size_t valid_points = 0;
    for (int v = 0; v < depth.rows; ++v) {
        for (int u = 0; u < depth.cols; ++u) {
            float depth_mm = 0.f;
            if (!ReadDepthMillimeters(depth, v, u, depth_mm)) continue;

            Eigen::Vector4f camera_point;
            camera_point << (static_cast<float>(u) - cam.cx) * depth_mm / cam.fx,
                            (static_cast<float>(v) - cam.cy) * depth_mm / cam.fy,
                            depth_mm,
                            1.f;
            Eigen::Vector3f point = (camera_to_world * camera_point).head<3>() * 1e-3f;
            if (use_geometry_transform)
                point = geometry_rotation * point + geometry_translation;
            for (size_t d = 0; d < 3; ++d) {
                scene_min[d] = std::min(scene_min[d], point[d]);
                scene_max[d] = std::max(scene_max[d], point[d]);
            }
            ++valid_points;
        }
    }

    return valid_points > 0 &&
           scene_max[0] > scene_min[0] &&
           scene_max[1] > scene_min[1] &&
           scene_max[2] > scene_min[2];
}

bool GraphPredictor::GetTeleNetworkGeometryTransform(Eigen::Matrix3f &rotation,
                                                     Eigen::Vector3f &translation) {
    rotation.setIdentity();
    translation.setZero();
    if (!mConfigPslam || !mConfigPslam->tele_network_geometry_compatibility)
        return false;

    Eigen::Matrix4f slam_world_to_camera;
    Eigen::Matrix4f network_camera_to_world;
    bool has_visual_frame = false;
    bool has_network_pose = false;
    {
        std::unique_lock<std::mutex> lock(mMutVisualFrame);
        has_visual_frame = mHasVisualFrame;
        has_network_pose = mHasTeleNetworkPose;
        slam_world_to_camera = mVisualPose;
        network_camera_to_world = mTeleNetworkPose;
    }
    if (!has_visual_frame)
        return false;

    const Eigen::Matrix4f slam_camera_to_world = slam_world_to_camera.inverse();
    const Eigen::Matrix3f slam_rotation = slam_camera_to_world.block<3, 3>(0, 0);
    const Eigen::Vector3f slam_translation = slam_camera_to_world.block<3, 1>(0, 3) * 1e-3f;

    Eigen::Matrix3f network_rotation;
    Eigen::Vector3f network_translation;
    if (has_network_pose) {
        network_rotation = network_camera_to_world.block<3, 3>(0, 0);
        network_translation = network_camera_to_world.block<3, 1>(0, 3) * 1e-3f;
    } else {
        Eigen::Matrix3f camera_basis_flip = Eigen::Matrix3f::Identity();
        camera_basis_flip(1, 1) = -1.0f;
        camera_basis_flip(2, 2) = -1.0f;
        network_rotation = slam_rotation * camera_basis_flip;
        network_translation = slam_translation;
    }

    rotation = network_rotation * slam_rotation.transpose();
    translation = network_translation - rotation * slam_translation;
    return true;
}

void GraphPredictor::Stop() {
    if(!mbThread) return;
    mbShouldStop=true;
    if (mThread.valid()) {
        while (mThread.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready)
            mThread.wait_for(std::chrono::milliseconds(50));
        mThread.get();
    }
}

void GraphPredictor::Process_IMPL() {
    bool runOnce=!mbThread;
    bool should_break = false;
    while (!mbShouldStop) {
        if(should_break) break;
        if(runOnce)should_break=true;
        /// Check if there are nodes that need to be updated
        std::map<int, NodePtr> vNodes;

        {
            // sleep this thread for a short time to prevnt continuous lock
            std::this_thread::sleep_for(std::chrono::milliseconds(50));

            std::unique_lock<std::mutex> lock(mpGraph->mMutNode);
            for (const auto &node : mpGraph->nodes) {
                if (!node.second->mbNeedUpdateNodeFeature) continue;
                vNodes.insert(node);
            }
        }
        if (vNodes.empty()) continue;

        if (mUseTeleModules) {
            // Incremental: only process nodes updated this frame (vNodes).
            // Historical nodes keep their predictions unchanged.
            // Edges are only built between objects co-detected in the current frame.
            std::map<int, NodePtr> selected_nodes;
            for (const auto &pair : vNodes) {
                if (pair.first == 0) continue;
                if (pair.second->surfels.size() < static_cast<size_t>(mConfigPslam->filter_num_node)) continue;
                selected_nodes.insert(pair);
            }

            std::map<int, std::pair<size_t, size_t>> sizeAndEdge;
            for (const auto &pair : selected_nodes) {
                sizeAndEdge[pair.first].first  = pair.second->surfels.size();
                sizeAndEdge[pair.first].second = selected_nodes.size() > 1 ? selected_nodes.size() - 1 : 0;
            }

            CTICK("[GraphPredictor][Tele]ProcessGraph");
            ProcessTele_IMPL(selected_nodes, sizeAndEdge);
            CTOCK("[GraphPredictor][Tele]ProcessGraph");
            mUpdated = true;
            continue;
        }

        size_t levels = GetParams().at("n_layers").int_value()+1; // 0: geometric feature, 1: gcn1 2: gcn2
        std::vector<std::map<int, NodePtr>> selected_nodes_level(levels);
        selected_nodes_level.at(0) = vNodes;
        for (size_t l = 1; l < levels; ++l) {
            auto nns = getNeighbor(selected_nodes_level.at(l-1), mConfigPslam->filter_num_node);
            for(auto &nn : nns)
                selected_nodes_level.at(l)[nn.first] = nn.second;
            for(auto &nn : selected_nodes_level.at(l-1))
                selected_nodes_level.at(l)[nn.first] = nn.second;
        }
        /// Save the current size and edge of all levels
        std::map<int,std::pair<size_t,size_t>> sizeAndEdge;
        {
            const auto &last_level = selected_nodes_level.back();
            for(const auto &pair : last_level) {
                auto i = pair.first;
                const auto &node = pair.second;
                sizeAndEdge[i].first = node->surfels.size();
                sizeAndEdge[i].second=0;
                std::unique_lock<std::mutex> lock(node->mMutNode);
                for (const auto &nn : node->neighbors) {
                    if(last_level.find(nn) == last_level.end()) continue;
                    sizeAndEdge.at(i).second += 1;
                }
            }
        }

        /// Update node feature
        SCLOG(VERBOSE) << "UpdateNodeFeature";
        CTICK("[Process_IMPL]1.updateNodeFeature");
        TICK("[Process_IMPL]1.updateNodeFeature");
        UpdateNodeFeature(selected_nodes_level.at(0));
//        UpdateNodeFeature(selected_nodes_level.back());
        TOCK("[Process_IMPL]1.updateNodeFeature");
        CTOCK("[Process_IMPL]1.updateNodeFeature");

        /// Update edge feature
        SCLOG(VERBOSE) << "UpdateEdgeFeature";
        CTICK("[Process_IMPL]2.updateEdgeFeature");
        TICK("[Process_IMPL]2.updateEdgeFeature");
        UpdateEdgeFeature(selected_nodes_level.at(1));
//        UpdateEdgeFeature(selected_nodes_level.back());
        TOCK("[Process_IMPL]2.updateEdgeFeature");
        CTOCK("[Process_IMPL]2.updateEdgeFeature");

        /// Update GCN feature
        SCLOG(VERBOSE) << "updateGCNFeature";

        if (GetParams().at("USE_GCN").bool_value()) {
            for (size_t l = 0; l < size_t(GetParams().at("n_layers").int_value()); ++l) {
                CTICK("[Process_IMPL]3.updateGCNFeature"+std::to_string(l));
                TICK("[Process_IMPL]3.updateGCNFeature"+std::to_string(l));
                UpdateGCNFeature(selected_nodes_level.at(l + 1), l);
//                UpdateGCNFeature(selected_nodes_level.back(), l);
                TOCK("[Process_IMPL]3.updateGCNFeature"+std::to_string(l));
                CTOCK("[Process_IMPL]3.updateGCNFeature"+std::to_string(l));
            }
        }


        /// Predict only the selected nodes
        SCLOG(VERBOSE) << "UpdatePrediction";
        CTICK("[Process_IMPL]4.updatePrediction");
        TICK("[Process_IMPL]4.updatePrediction");
        UpdatePrediction(selected_nodes_level.back(),sizeAndEdge);
        TOCK("[Process_IMPL]4.updatePrediction");
        CTOCK("[Process_IMPL]4.updatePrediction");

        mUpdated=true;
    }
    SCLOG(VERBOSE) << "stoped!";
}

void GraphPredictor::UpdateNodeFeature(const std::map<int,NodePtr> &vNodes) {
    SCLOG(VERBOSE) << "node feature start";
    auto n_nodes = vNodes.size();
    size_t n_pts = mConfigPslam->n_pts;

    size_t d_pts = 3;
    if (GetParams().at("USE_RGB").bool_value()) d_pts += 3;
    if (GetParams().at("USE_NORMAL").bool_value()) d_pts += 3;
    size_t d_edge = GetParams().at("dim_edge").int_value();

    /// Compute Node Shape Feature
    //// build input
    MemoryBlock3D input_nodes(MemoryBlock::DATA_TYPE::FLOAT,
                              {n_nodes, d_pts, n_pts});
    float max_dist, dist;
    CTICK("[UpdateNodeFeature]1.computeAndCopy");
    SCLOG(VERBOSE) << "computeAndCopy";
    size_t n=0;
    std::vector<SurfelPtr> selected_surfels;
    for (const auto &pair: vNodes) {
        auto &node = pair.second;
        const auto &centroid = node->mCentroid * 1e-3;
        max_dist = 0;
        {

            {
                std::unique_lock<std::mutex> lock (node->mMutSelected);
                if (node->selected_surfels.size() < n_pts) {
                    continue;
                    SCLOG(ERROR) << "";
                }
//                n_pts = selected_surfels.size();
                selected_surfels = node->selected_surfels;
            }


            const auto &surfels = selected_surfels;
            CTICK("[UpdateNodeFeature]1.1.copySurfels");
            try
            {
                for (size_t p = 0; p < n_pts; ++p) {
                    const auto &surfel = surfels.at(p);
                    dist = 0;
                    for (size_t d = 0; d < 3; ++d) {
                        input_nodes.at<float>(n, d, p) = surfel->pos[d] / 1000.f - centroid[d];
                        input_nodes.at<float>(n, d + 3, p) = float(surfel->color[2 - d]) / 255.f * 2.f - 1.f;
                        input_nodes.at<float>(n, d + 6, p) = surfel->normal[d];
                        dist += std::pow(input_nodes.at<float>(n, d, p), 2);
                    }
                    if (dist > max_dist) max_dist = dist;
                }
            }
            catch (std::out_of_range& e)
            {
                SCLOG(ERROR) << e.what();
            }

            CTOCK("[UpdateNodeFeature]1.1.copySurfels");
        }

        CTICK("[UpdateNodeFeature]1.2.normPts");
        max_dist = std::sqrt(max_dist);
        for (size_t p = 0; p < n_pts; ++p)
            for (size_t d = 0; d < 3; ++d)
                input_nodes.at<float>(n, d, p) /= max_dist;
        CTOCK("[UpdateNodeFeature]1.2.normPts");


        if(mDebug)
            ONNX::PrintVector<float,size_t>("input_nodes", input_nodes.Get<float>(n,0,0),
                                            {d_pts, n_pts});

        n++;
    }
    CTOCK("[UpdateNodeFeature]1.computeAndCopy");

    MemoryBlock2D descriptor_full(MemoryBlock::DATA_TYPE::FLOAT, {n_nodes, d_edge});
    {
        CTICK("[UpdateNodeFeature]2.collectInfoFromNodes");
        SCLOG(VERBOSE) << "collectInfoFromNodes";
        size_t i=0;
        for(const auto &pair : vNodes) {
//        for(size_t i=0;i<n_nodes;++i) {
            const auto &node = pair.second;
            const auto centroid = node->mCentroid*1e-3;
            const auto stdev    = node->mStd*1e-3;
            const auto bbox_min = node->mBBox_min*1e-3;
            const auto bbox_max = node->mBBox_max*1e-3;

            Eigen::Vector3f dim = (bbox_max-bbox_min);
            float volume = 1;
            for (size_t d = 0; d < 3; ++d) volume *= dim[d];
            float length = *std::max_element(dim.begin(), dim.end());

            for (size_t d = 0; d < 3; ++d) {
                descriptor_full.Row<float>(i)[d] = centroid[d];
                descriptor_full.Row<float>(i)[d + 3] = stdev[d];
                descriptor_full.Row<float>(i)[d + 6] = dim[d];
            }
            descriptor_full.Row<float>(i)[9] = volume;
            descriptor_full.Row<float>(i)[10] = length;
            i++;
        }
        CTOCK("[UpdateNodeFeature]2.collectInfoFromNodes");
    }

    if(mDebug)
        ONNX::PrintVector<float,size_t>("descriptor_full", descriptor_full.Get<float>(),
                                         {descriptor_full.mDims.x,descriptor_full.mDims.y});

    CTICK("[UpdateNodeFeature]3.compute");
    SCLOG(VERBOSE) << "compute";
    std::vector<float *> obj_inputs = {reinterpret_cast<float *>(input_nodes.data())};
    std::vector<std::vector<int64_t>> obj_input_sizes = {{static_cast<long>(n_nodes), static_cast<long>(d_pts), static_cast<long>(n_pts)}};
    auto out_enc_obj = mEatGCN->Compute(PSLAM::EatGCN::OP_ENC_OBJ, obj_inputs, obj_input_sizes);
    size_t dim_obj_feature = out_enc_obj[0].GetTensorTypeAndShapeInfo().GetShape()[1];
    auto out = ConcatObjFeature(n_nodes, dim_obj_feature, out_enc_obj[0], descriptor_full);
    CTOCK("[UpdateNodeFeature]3.compute");

    if(mDebug)
        ONNX::PrintVector<float,int64_t>("nodeFeature", out_enc_obj[0].GetTensorMutableData<float>(),
                {out_enc_obj[0].GetTensorTypeAndShapeInfo().GetShape()[0], out_enc_obj[0].GetTensorTypeAndShapeInfo().GetShape()[1]});

    /// Save Obj and Rel feature to nodes
    CTICK("[UpdateNodeFeature]4.UpdateNodeFeature");
    SCLOG(VERBOSE) << "UpdateNodeFeature";
    {
        size_t size_node_feature = out.mDims.y;
        size_t i = 0;
        for (const auto &pair : vNodes) {
//    for (size_t i=0; i < n_nodes; ++i) {
            auto &node = pair.second;
            auto &node_feature = node->mFeatures["0"];
            if (!node_feature) node_feature.reset(new MemoryBlock(MemoryBlock::FLOAT, size_node_feature));
            auto *feature = out.Row<float>(i);
            memcpy(node_feature->Get<float>(), feature, sizeof(float) * size_node_feature);
            node->mbNeedUpdateNodeFeature = false;
            i++;
        }
    }
    CTOCK("[UpdateNodeFeature]4.UpdateNodeFeature");



    SCLOG(VERBOSE) << "node feature end";
}

void GraphPredictor::UpdateEdgeFeature(const std::map<int,NodePtr> &vNodes){
    SCLOG(VERBOSE) << "edge feature start";
    // Build input for edges
//    const auto &nodes = mpGraph->nodes;
    size_t d_edge = GetParams().at("dim_edge").int_value();
    std::map<size_t, size_t> idx2seg;
    std::map<size_t, size_t> seg2idx;

    /// Collect the neighbor of the selected nodes
//    std::set<int> selected_nodes_idx;
//    for(const auto &node : vNodes) selected_nodes_idx.insert(node->idx);

    /// Build descriptor for these nodes
    SCLOG(VERBOSE) << "Build descriptor for these nodes";
    MemoryBlock2D descriptor_full(MemoryBlock::DATA_TYPE::FLOAT, {vNodes.size(), d_edge});
    {
        CTICK("[UpdateEdgeFeature]1.collectInfoFromNodes");
        size_t i=0;
        for(const auto &pair : vNodes) {
            const auto &node = pair.second;
            idx2seg[i] = node->idx;
            seg2idx[node->idx] = i;

            const auto centroid = node->mCentroid * 1e-3;
            const auto stdev = node->mStd * 1e-3;
            const auto bbox_min = node->mBBox_min * 1e-3;
            const auto bbox_max = node->mBBox_max * 1e-3;

            Eigen::Vector3f dim = (bbox_max - bbox_min);
            float volume = 1;
            for (size_t d = 0; d < 3; ++d) volume *= dim[d];
            float length = *std::max_element(dim.begin(), dim.end());

            for (size_t d = 0; d < 3; ++d) {
                descriptor_full.Row<float>(i)[d] = centroid[d];
                descriptor_full.Row<float>(i)[d + 3] = stdev[d];
                descriptor_full.Row<float>(i)[d + 6] = dim[d];
            }
            descriptor_full.Row<float>(i)[9] = volume;
            descriptor_full.Row<float>(i)[10] = length;

            if(mDebug) {
                std::stringstream ss;//TODO: remove me
                ss << "[" << node->idx << "] ";
                for (size_t j = 0; j < d_edge; ++j)
                    ss << descriptor_full.Row<float>(i)[j] << " ";
                SCLOG(DEBUG) << ss.str();
            }

            i++;
        }
        CTOCK("[UpdateEdgeFeature]1.collectInfoFromNodes");
    }

    /// Count Edges
    CTICK("[UpdateEdgeFeature]2.countEdges");
    SCLOG(VERBOSE) << "countEdges";
    std::vector<std::pair<int,int>> selected_edges;
    {
//        std::unique_lock<std::mutex> lock(mpGraph->mMutEdge);
        for(const auto & pair : vNodes) {
            const auto &n = pair.first;
            std::unique_lock<std::mutex> lock(pair.second->mMutNN);
            for (const auto &nn : pair.second->neighbors) {
                if (vNodes.find(nn) == vNodes.end()) continue;
                if (seg2idx.find(nn) == seg2idx.end()) continue;
                if (seg2idx.at(n) == seg2idx.at(nn)) continue;
                selected_edges.emplace_back(seg2idx.at(n),seg2idx.at(nn));
            }
        }
    }
    size_t n_edges = selected_edges.size();
    CTOCK("[UpdateEdgeFeature]2.countEdges");

    if(mDebug)
    {//TODO: remove me
        std::stringstream ss;
        ss << "edges:\n";
        for(auto &pair:selected_edges)
            ss << pair.first << ", " << pair.second << "\n";

        SCLOG(DEBUG) << ss.str();
    }


    /// Compute Edge descriptor
    CTICK("[UpdateEdgeFeature]3.buildEdgeIndex");
    SCLOG(VERBOSE) << "buildEdgeIndex";
    std::map<int, std::pair<size_t, size_t>> nodeSizeEdgeTracker;
    MemoryBlock2D edge_index(MemoryBlock2D::INT64_T, { 2, n_edges });
    for(size_t i=0;i<selected_edges.size();++i) {
        edge_index.at<int64_t>(0, i) = selected_edges.at(i).first;
        edge_index.at<int64_t>(1, i) = selected_edges.at(i).second;
    }
    CTOCK("[UpdateEdgeFeature]3.buildEdgeIndex");

    /// Find Nodes and Edges feature
    CTICK("[UpdateEdgeFeature]4.computeEdgeFeature");
    SCLOG(VERBOSE) << "computeEdgeFeature";
    auto edge_descriptor = DataUtil::compute_edge_descriptor(descriptor_full, edge_index, source_to_target);
    std::vector<float *> rel_inputs = {static_cast<float *>(edge_descriptor.data())};
    std::vector<std::vector<int64_t>> rel_input_sizes = {{static_cast<long>(n_edges), static_cast<long>(d_edge), 1}};
    auto out_enc_rel = mEatGCN->Compute(PSLAM::EatGCN::OP_ENC_REL, rel_inputs, rel_input_sizes);
    CTOCK("[UpdateEdgeFeature]4.computeEdgeFeature");

    //TODO: remove me
    if(mDebug)
    {
        ONNX::PrintVector<float,size_t>("edge_descriptor",edge_descriptor.Get<float>(), {edge_descriptor.mDims.x,edge_descriptor.mDims.y});
        ONNX::PrintTensorShape("out_enc_rel",out_enc_rel[0]);
        ONNX::PrintVector<float>("out_enc_rel",out_enc_rel[0].GetTensorMutableData<float>(),
                std::vector<size_t>{static_cast<unsigned long>(out_enc_rel[0].GetTensorTypeAndShapeInfo().GetShape()[0]),
                 static_cast<unsigned long>(out_enc_rel[0].GetTensorTypeAndShapeInfo().GetShape()[1])});
    }


    CTICK("[UpdateEdgeFeature]5.copyEdegeFeature");
    SCLOG(VERBOSE) << "copyEdegeFeature";
    const auto &edges = mpGraph->edges;
    int64_t size_edge_feature = out_enc_rel[0].GetTensorTypeAndShapeInfo().GetShape()[1];
    for (size_t i=0; i < n_edges; ++i) {
        auto idx_from = idx2seg.at( edge_index.at<int64_t>(0,i) );
        auto idx_to = idx2seg.at( edge_index.at<int64_t>(1,i) );
        EdgePtr edge;
        {
            std::unique_lock<std::mutex> lock(mpGraph->mMutEdge);
            if (edges.find({idx_from,idx_to}) == edges.end()) {
                edge = std::make_shared<PSLAM::Edge>();
                edge->nodeFrom=idx_from;
                edge->nodeTo=idx_to;
            } else {
                edge = edges.at({idx_from,idx_to});
            }
        }




//        SCLOG(VERBOSE) << "get feature";
        auto &edge_feature = edge->mFeatures["0"];
        if (!edge_feature) edge_feature.reset(new MemoryBlock(MemoryBlock::FLOAT, size_edge_feature));
        float *feature = out_enc_rel[0].GetTensorMutableData<float>()+i*size_edge_feature;
        memcpy(edge_feature->Get<float>(),feature,sizeof(float)*size_edge_feature);
        mpGraph->AddEdge(edge);
    }
    CTOCK("[UpdateEdgeFeature]5.copyEdegeFeature");

    SCLOG(VERBOSE) << "edge feature end";
}

void GraphPredictor::UpdateGCNFeature(const std::map<int,NodePtr> &vNodes, size_t level){
    SCLOG(VERBOSE) << "Update GCN " << level << " feature start";
    const auto &edges = mpGraph->edges;
    std::map<size_t, size_t> idx2seg;
    std::map<size_t, size_t> seg2idx;
    /// Filter out nodes without computed feature
//    std::vector<NodePtr> selected_nodes;
    std::map<int,NodePtr> selected_nodes;
    {
        for(const auto & node : vNodes){
            if(node.second->mFeatures.find(std::to_string(level)) != node.second->mFeatures.end())
                selected_nodes.insert(node);//.push_back(node);
        }
    }

    size_t n_nodes = selected_nodes.size();
    if(n_nodes==0){
        SCLOG(VERBOSE) << "got 0 nodes. return";
        return;
    }
    {
        size_t i=0;
        for (const auto &pair:selected_nodes) {
            idx2seg[i] = pair.second->idx;
            seg2idx[pair.second->idx] = i;
            i++;
        }
    }



    /// Count Edges
    SCLOG(VERBOSE) << "Update countEdges";
    CTICK("[UpdateGCNFeature]1.countEdges");
    std::vector<EdgePtr> selected_edges;
    {
        for(auto & selected_node : selected_nodes) {
            const auto &n = selected_node.first;
            if (seg2idx.find(n) == seg2idx.end()) continue;
            std::unique_lock<std::mutex> lock(selected_node.second->mMutNN);
            for (const auto &nn : selected_node.second->neighbors) {
                if (selected_nodes.find(nn) == selected_nodes.end()) continue;
                if (seg2idx.find(nn) == seg2idx.end()) continue;
                if (seg2idx.at(n) == seg2idx.at(nn)) continue;
                std::unique_lock<std::mutex> lock(mpGraph->mMutEdge);
                if(edges.find({n,nn}) == edges.end()) continue;
                if(edges.at({n,nn})->mFeatures.find(std::to_string(level)) ==
                        edges.at({n,nn})->mFeatures.end()) continue;
                selected_edges.push_back(edges.at({n,nn}));
            }
        }
    }
    size_t n_edges = selected_edges.size();
    CTOCK("[UpdateGCNFeature]1.countEdges");
    if(n_edges==0){
        SCLOG(VERBOSE) << "got 0 edges. return";
        return;
    }

    /// Compute Edge descriptor
    SCLOG(VERBOSE) << "buildEdgeIndex";
    CTICK("[UpdateGCNFeature]2.buildEdgeIndex");
    std::map<int, std::pair<size_t, size_t>> nodeSizeEdgeTracker;
    MemoryBlock2D edge_index(MemoryBlock2D::INT64_T, { 2, n_edges });
    for(size_t i=0;i<selected_edges.size();++i) {
        edge_index.at<int64_t>(0, i) = seg2idx.at(selected_edges.at(i)->nodeFrom);
        edge_index.at<int64_t>(1, i) = seg2idx.at(selected_edges.at(i)->nodeTo);
    }
    CTOCK("[UpdateGCNFeature]2.buildEdgeIndex");

    if(mDebug){
        std::stringstream ss;
        ss << "edges\n";
        for(size_t i=0;i<n_edges;++i)
            ss << edge_index.at<int64>(0,i) << ", " << edge_index.at<int64_t>(1,i) << "\n";
        SCLOG(DEBUG) << ss.str();
    }


    /// Build Node Input
    int64_t dim_obj_f = GetParams().at("dim_o_f").int_value();
    MemoryBlock2D nodeFeatures(MemoryBlock::FLOAT, {selected_nodes.size(), static_cast<unsigned long>(dim_obj_f)});
    SCLOG(VERBOSE) << "copyNodeFeature";
    CTICK("[UpdateGCNFeature]3.copyNodeFeature");
    {
        size_t i=0;
        for(const auto &pair:selected_nodes){
            auto &node = pair.second;
            const auto *feature = node->mFeatures.at(std::to_string(level))->Get<float>();
            memcpy(nodeFeatures.Row<float>(i),feature,sizeof(float)*dim_obj_f);
            i++;
        }
    }
    CTOCK("[UpdateGCNFeature]3.copyNodeFeature");

    if(mDebug)
    ONNX::PrintVector<float,unsigned long>("nodeFeature", nodeFeatures.Get<float>(), {selected_nodes.size(), static_cast<unsigned long>(dim_obj_f)});

    /// Build Edge Input
    int64_t dim_rel_f = GetParams().at("dim_r_f").int_value();
    MemoryBlock2D edgeFeatures(MemoryBlock::FLOAT, {n_edges, static_cast<unsigned long>(dim_rel_f)});
    SCLOG(VERBOSE) << "copyEdgeFeature";
    CTICK("[UpdateGCNFeature]4.copyEdgeFeature");
    for(size_t i=0;i<n_edges;++i){
        auto &edge = selected_edges.at(i);
        const auto *feature = edge->mFeatures.at( std::to_string(level) )->Get<float>();
        memcpy(edgeFeatures.Row<float>(i),feature,sizeof(float)*dim_rel_f);

        if(mDebug)
        {//TODO: remove me
            std::stringstream ss;
            ss.precision(4);
            ss << "["<<edge->nodeFrom<<","<<edge->nodeTo<<"]:";
            for(size_t j=0;j<edge->mFeatures.at( std::to_string(level) )->size();++j) ss << feature[j] << " ";
            SCLOG(DEBUG) << ss.str();
        }
        if(mDebug)
        {//TODO: remove me
            std::stringstream ss;
            ss.precision(4);
            ss << "["<<edge->nodeFrom<<","<<edge->nodeTo<<"]:";
            for(size_t j=0;j<edge->mFeatures.at( std::to_string(level) )->size();++j) ss << edgeFeatures.Row<float>(i)[j] << " ";
            SCLOG(DEBUG) << ss.str();
        }
    }
    CTOCK("[UpdateGCNFeature]4.copyEdgeFeature");

    if(mDebug)
    ONNX::PrintVector<float,unsigned long>("edgeFeatures", edgeFeatures.Get<float>(), {n_edges, static_cast<unsigned long>(dim_rel_f)});

    /// Compute
    SCLOG(VERBOSE) << "compute";
    CTICK("[UpdateGCNFeature]5.compute");
    {
        size_t i_i = source_to_target ? 1 : 0;
        size_t i_j = source_to_target ? 0 : 1;

        std::vector<float> obj_f_i = DataUtil::Collect(nodeFeatures.Get<float>(),
                                                       edge_index.Row<int64_t>(i_i), dim_obj_f, n_edges);
        std::vector<float> obj_f_j = DataUtil::Collect(nodeFeatures.Get<float>(),
                                                       edge_index.Row<int64_t>(i_j), dim_obj_f, n_edges);
        if(mDebug){
            ONNX::PrintVector("obj_f_i", obj_f_i.data(),
                        std::vector<int64_t>{static_cast<long>(n_edges), dim_obj_f});
            ONNX::PrintVector("obj_f_j", obj_f_j.data(),
                        std::vector<int64_t>{static_cast<long>(n_edges), dim_obj_f});
        }

        std::vector<Ort::Value> inputs;
        inputs.emplace_back(ONNX::CreateTensor(mMemoryInfo.get(), obj_f_i.data(),
                {static_cast<long>(n_edges), dim_obj_f}));
        inputs.emplace_back(ONNX::CreateTensor(mMemoryInfo.get(), edgeFeatures.Get<float>(),
                {static_cast<long>(n_edges), dim_rel_f}));
        inputs.emplace_back(ONNX::CreateTensor(mMemoryInfo.get(), obj_f_j.data(),
                {static_cast<long>(n_edges), dim_obj_f}));

        auto output_atten = mEatGCN->ComputeGCN(PSLAM::EatGCN::OP_GCN_ATTEN, level, inputs);
        if (mDebug) {
            ONNX::PrintVector("output_atten", output_atten[0].GetTensorMutableData<float>(),
                        std::vector<int64_t>{output_atten[0].GetTensorTypeAndShapeInfo().GetShape()[0],
                                             output_atten[0].GetTensorTypeAndShapeInfo().GetShape()[1]});
            ONNX::PrintVector("output_edge", output_atten[1].GetTensorMutableData<float>(),
                        std::vector<int64_t>{output_atten[1].GetTensorTypeAndShapeInfo().GetShape()[0],
                                             output_atten[1].GetTensorTypeAndShapeInfo().GetShape()[1]});
        }

        auto dim_hidden = output_atten[0].GetTensorTypeAndShapeInfo().GetShape()[1];
        auto xx = DataUtil::IndexAggr(output_atten[0].GetTensorMutableData<float>(),
                                      edge_index.Row<int64_t>(i_i), dim_hidden, n_edges, n_nodes, 1);
//        if (bVerbose) PrintVector("xx_aggr", xx.data(), std::vector<int64_t>{n_node, dim_hidden});

        auto xxx = DataUtil::Concat(nodeFeatures.Get<float>(), xx.data(), n_nodes,
                                    dim_obj_f, dim_hidden,
                                    dim_obj_f, dim_hidden);
        if (mDebug) ONNX::PrintVector("xxx_concat", xxx.data(), std::vector<int64_t>{static_cast<long>(n_nodes), dim_obj_f + dim_hidden});

        std::vector<float *> gcn_nn_inputs = {xxx.data()};
        std::vector<std::vector<int64_t>> gcn_nn_input_size = {{static_cast<long>(n_nodes), dim_obj_f + dim_hidden}};
        auto gcn_nn2_outputs = mEatGCN->ComputeGCN(PSLAM::EatGCN::OP_GCN_PROP, level, gcn_nn_inputs, gcn_nn_input_size);
        if (level + 1 < size_t(GetParams().at("n_layers").int_value())) {
            DataUtil::Relu(gcn_nn2_outputs[0].GetTensorMutableData<float>(), n_nodes * dim_obj_f);
            DataUtil::Relu(output_atten[1].GetTensorMutableData<float>(), n_edges * dim_rel_f);
        }

        /// Copy Node feature
        {
            size_t n=0;
            for(const auto & pair : selected_nodes) {
                auto& node_feature = pair.second->mFeatures[std::to_string(level+1)];
                if(!node_feature) node_feature.reset(new MemoryBlock(MemoryBlock::FLOAT, dim_obj_f));
                auto *feature = gcn_nn2_outputs[0].GetTensorMutableData<float>() + n * dim_obj_f;
                memcpy(node_feature->Get<float>(), feature, sizeof(float) * dim_obj_f);
                n++;
            }
        }


        /// Copy Edge feature
        for(size_t n=0;n<n_edges;++n) {
            auto& edge_feature = selected_edges.at(n)->mFeatures[std::to_string(level+1)];
            if(!edge_feature) edge_feature.reset(new MemoryBlock(MemoryBlock::FLOAT, dim_rel_f));
            auto *feature = output_atten[1].GetTensorMutableData<float>() + n * dim_rel_f;
            memcpy(edge_feature->Get<float>(), feature, sizeof(float) * dim_rel_f);
        }
    }
    CTOCK("[UpdateGCNFeature]5.compute");
    SCLOG(VERBOSE) << "Update GCN " << level << " feature done";
}


std::map<int,NodePtr> GraphPredictor::getNeighbor(const std::map<int, NodePtr> &vNodes, size_t min_pts) {
    /// Collect the neighbors of the selected nodes that need to be updated
    std::map<int,NodePtr> nodes;
    std::set<int> all_selected_nodes;
    std::vector<std::set<int>> level_selected_indices;
    {
        CTICK("[UpdateShapeFeature]2.findNeighbors");
        std::set<int> selected_nodes;
        for (const auto &n : vNodes)
            selected_nodes.insert(n.second->idx);
        level_selected_indices.push_back(selected_nodes);
        for (size_t l = 1; l < 2; ++l) {
            std::set<int> nn_selections;
            for (auto &s_n : level_selected_indices.at(l - 1)) {
//                    if (nodes.at(s_n)->last_class_predicted == "floor")continue;
                std::unique_lock<std::mutex> lock(mpGraph->mMutNode);
                if (mpGraph->nodes.find(s_n) == mpGraph->nodes.end()) continue;
                std::unique_lock<std::mutex> lock2(mpGraph->nodes.at(s_n)->mMutNN);
                for (auto &nn : mpGraph->nodes.at(s_n)->neighbors) {
                    if (mpGraph->nodes.find(nn) == mpGraph->nodes.end()) continue;
                    if (mpGraph->nodes.at(nn)->surfels.size() <= min_pts) continue;
                    if(all_selected_nodes.find(nn) != all_selected_nodes.end())continue;
                    nn_selections.insert(nn);
                    nodes[nn] = mpGraph->nodes.at(nn);
                    all_selected_nodes.insert(nn);
                }
            }
            level_selected_indices.push_back(nn_selections);
        }
        CTOCK("[UpdateShapeFeature]2.findNeighbors");
    }

    return nodes;
}

MemoryBlock2D GraphPredictor::BuildTeleVisualFeature(const std::vector<NodePtr> &nodes,
                                                     size_t visual_dim) {
    MemoryBlock2D visual_feature(MemoryBlock::FLOAT, {nodes.size(), visual_dim});
    visual_feature.Clear();
    if (!mUseTeleVisualEncoder || nodes.empty()) return visual_feature;

    cv::Mat rgb;
    Eigen::Matrix4f pose;
    CameraParameters cam{};
    int frame_idx = -1;
    {
        std::unique_lock<std::mutex> lock(mMutVisualFrame);
        if (!mHasVisualFrame || mVisualRGB.empty()) return visual_feature;
        rgb = mVisualRGB.clone();
        pose = mVisualPose;
        cam = mVisualCam;
        frame_idx = mVisualFrameIndex;
    }
    if (cam.width == 0 || cam.height == 0) return visual_feature;

    const int crop_size = 224;
    const std::string debug_dir = mConfigPslam ? WithTrailingSlash(mConfigPslam->tele_debug_dir) : "";
    struct VisualDebugRecord {
        int frame_idx = -1;
        int node_idx = -1;
        size_t count = 0;
        float x1 = 0.f, y1 = 0.f, x2 = 0.f, y2 = 0.f;
        int left = 0, top = 0, right = 0, bottom = 0;
        std::string crop_path;
    };
    std::vector<VisualDebugRecord> debug_records;
    cv::Mat debug_overlay;
    if (!debug_dir.empty()) debug_overlay = rgb.clone();

    std::vector<size_t> valid_indices;
    std::vector<cv::Mat> crops;
    valid_indices.reserve(nodes.size());
    crops.reserve(nodes.size());

    const Eigen::Matrix4f inverse_pose = pose.inverse();
    for (size_t i = 0; i < nodes.size(); ++i) {
        float x1a, y1a, x2a, y2a;
        float x1b, y1b, x2b, y2b;
        const size_t count_a = ProjectSurfelBBox(nodes.at(i), pose, cam, x1a, y1a, x2a, y2a);
        const size_t count_b = ProjectSurfelBBox(nodes.at(i), inverse_pose, cam, x1b, y1b, x2b, y2b);

        float x1 = x1a, y1 = y1a, x2 = x2a, y2 = y2a;
        size_t count = count_a;
        if (count_b > count_a) {
            x1 = x1b;
            y1 = y1b;
            x2 = x2b;
            y2 = y2b;
            count = count_b;
        }
        if (count == 0 || x2 <= x1 || y2 <= y1) continue;

        const int left = std::max(0, static_cast<int>(std::floor(x1)));
        const int top = std::max(0, static_cast<int>(std::floor(y1)));
        const int right = std::min(rgb.cols, static_cast<int>(std::ceil(x2)) + 1);
        const int bottom = std::min(rgb.rows, static_cast<int>(std::ceil(y2)) + 1);
        if (right <= left || bottom <= top) continue;

        cv::Mat crop = rgb(cv::Rect(left, top, right - left, bottom - top)).clone();
        cv::resize(crop, crop, cv::Size(crop_size, crop_size), 0, 0, cv::INTER_LINEAR);

        if (!debug_dir.empty()) {
            std::ostringstream crop_name;
            crop_name << "frame_" << std::setw(6) << std::setfill('0') << frame_idx
                      << "_node_" << nodes.at(i)->idx << "_crop.jpg";
            cv::Mat crop_bgr;
            cv::cvtColor(crop, crop_bgr, cv::COLOR_RGB2BGR);
            cv::imwrite(debug_dir + crop_name.str(), crop_bgr);

            cv::rectangle(debug_overlay, cv::Rect(left, top, right - left, bottom - top),
                          cv::Scalar(255, 0, 0), 2);
            cv::putText(debug_overlay, std::to_string(nodes.at(i)->idx),
                        cv::Point(left, std::max(12, top - 4)),
                        cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(255, 0, 0), 1);

            VisualDebugRecord record;
            record.frame_idx = frame_idx;
            record.node_idx = nodes.at(i)->idx;
            record.count = count;
            record.x1 = x1;
            record.y1 = y1;
            record.x2 = x2;
            record.y2 = y2;
            record.left = left;
            record.top = top;
            record.right = right;
            record.bottom = bottom;
            record.crop_path = crop_name.str();
            debug_records.emplace_back(record);
        }

        valid_indices.emplace_back(i);
        crops.emplace_back(std::move(crop));
    }

    if (valid_indices.empty()) return visual_feature;

    MemoryBlock3D input_crops(MemoryBlock::FLOAT,
                              {valid_indices.size(), 3, static_cast<size_t>(crop_size * crop_size)});
    for (size_t n = 0; n < valid_indices.size(); ++n) {
        cv::Mat crop_float;
        crops.at(n).convertTo(crop_float, CV_32FC3, 1.0 / 255.0);
        for (int y = 0; y < crop_size; ++y) {
            const auto *row = crop_float.ptr<cv::Vec3f>(y);
            for (int x = 0; x < crop_size; ++x) {
                const size_t p = static_cast<size_t>(y * crop_size + x);
                input_crops.at<float>(n, 0, p) = row[x][0];
                input_crops.at<float>(n, 1, p) = row[x][1];
                input_crops.at<float>(n, 2, p) = row[x][2];
            }
        }
    }

    std::vector<Ort::Value> inputs;
    inputs.emplace_back(ONNX::CreateTensor(mMemoryInfo.get(), input_crops.Get<float>(),
                                           {static_cast<int64_t>(valid_indices.size()), 3, crop_size, crop_size}));
    auto outputs = RunTeleModule("visual_encoder", inputs);
    auto valid_visual = CopyTensor2D(outputs[0]);
    const size_t copy_dim = std::min(visual_dim, valid_visual.mDims.y);
    for (size_t n = 0; n < valid_indices.size(); ++n) {
        std::memcpy(visual_feature.Row<float>(valid_indices.at(n)),
                    valid_visual.Row<float>(n),
                    sizeof(float) * copy_dim);
    }

    if (!debug_dir.empty()) {
        if (!debug_overlay.empty()) {
            std::ostringstream overlay_name;
            overlay_name << "frame_" << std::setw(6) << std::setfill('0') << frame_idx
                         << "_overlay.jpg";
            cv::Mat overlay_bgr;
            cv::cvtColor(debug_overlay, overlay_bgr, cv::COLOR_RGB2BGR);
            cv::imwrite(debug_dir + overlay_name.str(), overlay_bgr);
        }

        const std::string csv_path = debug_dir + "tele_visual_bbox.csv";
        const bool write_header = !FileExists(csv_path);
        std::fstream csv(csv_path, std::ios::app);
        if (write_header)
            csv << "frame,node,projected_points,x1,y1,x2,y2,left,top,right,bottom,visual_l2,crop\n";
        for (size_t n = 0; n < debug_records.size(); ++n) {
            const auto &record = debug_records.at(n);
            const size_t visual_idx = valid_indices.at(n);
            float norm2 = 0.f;
            for (size_t d = 0; d < copy_dim; ++d) {
                const float value = visual_feature.at<float>(visual_idx, d);
                norm2 += value * value;
            }
            csv << record.frame_idx << "," << record.node_idx << "," << record.count << ","
                << record.x1 << "," << record.y1 << "," << record.x2 << "," << record.y2 << ","
                << record.left << "," << record.top << "," << record.right << "," << record.bottom << ","
                << std::sqrt(norm2) << "," << record.crop_path << "\n";
        }
    }

    return visual_feature;
}

void GraphPredictor::ProcessTele_IMPL(const std::map<int,NodePtr> &selected_nodes,
                                      const std::map<int,std::pair<size_t,size_t>> &sizeAndEdge) {
    if (selected_nodes.empty()) return;

    const size_t n_pts = mConfigPslam->n_pts;
    std::map<int, NodePtr> ready_nodes;
    CTICK("[Tele]ready_nodes");
    for (const auto &pair : selected_nodes) {
        if (pair.first == 0) continue;
        auto &node = pair.second;
        bool ready = false;
        {
            std::unique_lock<std::mutex> lock(node->mMutSelected);
            ready = node->selected_surfels.size() == n_pts;
        }
        if (!ready && !node->mbNeedUpdateNodeFeature)
            node->UpdateSelectedNode(0, mConfigPslam->filter_num_node, n_pts, true);
        {
            std::unique_lock<std::mutex> lock(node->mMutSelected);
            ready = node->selected_surfels.size() == n_pts;
        }
        if (ready) ready_nodes.insert(pair);
    }
    CTOCK("[Tele]ready_nodes");

    if (ready_nodes.empty()) return;

    const size_t n_nodes = ready_nodes.size();
    const std::string debug_dir = mConfigPslam ? WithTrailingSlash(mConfigPslam->tele_debug_dir) : "";
    int debug_frame_idx = -1;
    if (!debug_dir.empty()) {
        std::unique_lock<std::mutex> lock(mMutVisualFrame);
        debug_frame_idx = mVisualFrameIndex;
    }

    std::vector<NodePtr> nodes;
    nodes.reserve(n_nodes);
    for (const auto &pair : ready_nodes)
        nodes.emplace_back(pair.second);

    MemoryBlock3D input_nodes(MemoryBlock::FLOAT, {n_nodes, 3, n_pts});
    MemoryBlock2D descriptor_full(MemoryBlock::FLOAT, {n_nodes, 11});
    MemoryBlock2D foreground_geo(MemoryBlock::FLOAT, {n_nodes, 9});
    MemoryBlock2D position_geo(MemoryBlock::FLOAT, {n_nodes, 4});
    position_geo.Clear();

    std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> full_centroids(n_nodes);
    std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> full_bbox_mins(n_nodes);
    std::vector<size_t> full_surfel_counts(n_nodes, 0);
    Eigen::Vector3f scene_min, scene_max;
    scene_min.setConstant(std::numeric_limits<float>::max());
    scene_max.setConstant(std::numeric_limits<float>::lowest());

    Eigen::Matrix3f tele_geometry_rotation = Eigen::Matrix3f::Identity();
    Eigen::Vector3f tele_geometry_translation = Eigen::Vector3f::Zero();
    const bool use_tele_geometry_compatibility =
            GetTeleNetworkGeometryTransform(tele_geometry_rotation, tele_geometry_translation);
    const auto network_geometry_point =
            [&](const Eigen::Vector3f &point) -> Eigen::Vector3f {
                if (!use_tele_geometry_compatibility) return point;
                return tele_geometry_rotation * point + tele_geometry_translation;
            };

    // Use object bounding boxes for scene bounds instead of scanning the full depth image.
    // ComputeTeleFrameSceneBounds iterated every depth pixel (~300k) per GNN call, costing ~380ms.
    bool use_frame_scene_bounds = false;

    bool use_full_map_scene_bounds = false;
    CTICK("[Tele]scene_bounds");
    if (!use_frame_scene_bounds && use_tele_geometry_compatibility && mpGraph) {
        size_t scene_bound_points = 0;
        std::vector<NodePtr> graph_nodes;
        {
            std::unique_lock<std::mutex> graph_lock(mpGraph->mMutNode);
            graph_nodes.reserve(mpGraph->nodes.size());
            for (const auto &node_pair : mpGraph->nodes) {
                if (node_pair.first == 0) continue;
                graph_nodes.emplace_back(node_pair.second);
            }
        }
        for (const auto &node : graph_nodes) {
            std::unique_lock<std::mutex> lock(node->mMutSurfel);
            for (const auto &surfel_pair : node->surfels) {
                const auto &surfel = surfel_pair.second;
                if (!surfel->is_stable || !surfel->is_valid) continue;
                const Eigen::Vector3f point = network_geometry_point(surfel->pos * 1e-3f);
                for (size_t d = 0; d < 3; ++d) {
                    scene_min[d] = std::min(scene_min[d], point[d]);
                    scene_max[d] = std::max(scene_max[d], point[d]);
                }
                ++scene_bound_points;
            }
        }
        use_full_map_scene_bounds = scene_bound_points > 0;
        if (use_full_map_scene_bounds)
            SCLOG(VERBOSE) << "Tele network geometry compatibility uses "
                           << scene_bound_points << " stable map surfels for scene bounds.";
    }
    CTOCK("[Tele]scene_bounds");

    CTICK("[Tele]0.cpu_preprocess");
    for (size_t n = 0; n < n_nodes; ++n) {
        const auto &node = nodes.at(n);
        std::vector<SurfelPtr> sampled_surfels;
        {
            std::unique_lock<std::mutex> lock(node->mMutSelected);
            sampled_surfels = node->selected_surfels;
        }
        if (sampled_surfels.size() != n_pts) continue;

        std::vector<SurfelPtr> full_surfels;
        {
            std::unique_lock<std::mutex> lock(node->mMutSurfel);
            full_surfels.reserve(node->surfels.size());
            for (const auto &surfel_pair : node->surfels)
                if (surfel_pair.second->is_stable && surfel_pair.second->is_valid)
                    full_surfels.emplace_back(surfel_pair.second);
        }
        if (full_surfels.empty()) full_surfels = sampled_surfels;

        Eigen::Vector3f sampled_centroid = Eigen::Vector3f::Zero();
        for (const auto &surfel : sampled_surfels)
            sampled_centroid += network_geometry_point(surfel->pos * 1e-3f);
        sampled_centroid /= static_cast<float>(sampled_surfels.size());

        Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
        Eigen::Vector3f bbox_min, bbox_max;
        bbox_min.setConstant(std::numeric_limits<float>::max());
        bbox_max.setConstant(std::numeric_limits<float>::lowest());
        for (const auto &surfel : full_surfels) {
            const Eigen::Vector3f point = network_geometry_point(surfel->pos * 1e-3f);
            centroid += point;
            for (size_t d = 0; d < 3; ++d) {
                bbox_min[d] = std::min(bbox_min[d], point[d]);
                bbox_max[d] = std::max(bbox_max[d], point[d]);
            }
        }
        centroid /= static_cast<float>(full_surfels.size());

        Eigen::Vector3f variance = Eigen::Vector3f::Zero();
        Eigen::Matrix3f cov = Eigen::Matrix3f::Zero();
        for (const auto &surfel : full_surfels) {
            const Eigen::Vector3f point = network_geometry_point(surfel->pos * 1e-3f);
            const Eigen::Vector3f centered = point - centroid;
            variance += centered.cwiseProduct(centered);
            cov += centered * centered.transpose();
        }
        Eigen::Vector3f stdev;
        for (size_t d = 0; d < 3; ++d)
            stdev[d] = std::sqrt(std::max(variance[d] / static_cast<float>(full_surfels.size()), 0.f));
        if (full_surfels.size() > 1) cov /= static_cast<float>(full_surfels.size() - 1);

        Eigen::Vector3f dims = bbox_max - bbox_min;
        for (size_t d = 0; d < 3; ++d) dims[d] = std::max(dims[d], 1e-6f);
        float volume = dims[0] * dims[1] * dims[2];
        float length = std::max(dims[0], std::max(dims[1], dims[2]));

        full_centroids[n] = centroid;
        full_bbox_mins[n] = bbox_min;
        full_surfel_counts[n] = full_surfels.size();
        for (size_t d = 0; d < 3; ++d) {
            if (!use_frame_scene_bounds && !use_full_map_scene_bounds) {
                scene_min[d] = std::min(scene_min[d], bbox_min[d]);
                scene_max[d] = std::max(scene_max[d], bbox_max[d]);
            }
            descriptor_full.at<float>(n, d) = centroid[d];
            descriptor_full.at<float>(n, d + 3) = stdev[d];
            descriptor_full.at<float>(n, d + 6) = dims[d];
        }
        descriptor_full.at<float>(n, 9) = std::max(volume, 1e-6f);
        descriptor_full.at<float>(n, 10) = std::max(length, 1e-6f);

        float max_dist = 0.f;
        for (size_t p = 0; p < n_pts; ++p) {
            const auto &surfel = sampled_surfels.at(p);
            const Eigen::Vector3f point = network_geometry_point(surfel->pos * 1e-3f);
            float dist = 0.f;
            for (size_t d = 0; d < 3; ++d) {
                input_nodes.at<float>(n, d, p) = point[d] - sampled_centroid[d];
                dist += input_nodes.at<float>(n, d, p) * input_nodes.at<float>(n, d, p);
            }
            max_dist = std::max(max_dist, dist);
        }
        max_dist = std::sqrt(max_dist);
        if (max_dist <= 1e-6f) max_dist = 1.f;
        for (size_t p = 0; p < n_pts; ++p)
            for (size_t d = 0; d < 3; ++d)
                input_nodes.at<float>(n, d, p) /= max_dist;

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(cov);
        Eigen::Vector3f eigvals = solver.eigenvalues();
        float eig0 = std::max(eigvals[2], 1e-8f);
        float eig1 = std::max(eigvals[1], 1e-8f);
        float eig2 = std::max(eigvals[0], 1e-8f);
        float planarity = (eig1 - eig2) / eig0;

        for (size_t d = 0; d < 3; ++d)
            foreground_geo.at<float>(n, d) = dims[d];
        foreground_geo.at<float>(n, 3) = centroid[1];
        foreground_geo.at<float>(n, 4) = std::log1p(static_cast<float>(full_surfels.size()));
        foreground_geo.at<float>(n, 5) = eig0;
        foreground_geo.at<float>(n, 6) = eig1;
        foreground_geo.at<float>(n, 7) = eig2;
        foreground_geo.at<float>(n, 8) = planarity;
    }

    Eigen::Vector3f scene_size = scene_max - scene_min;
    for (size_t d = 0; d < 3; ++d) scene_size[d] = std::max(scene_size[d], 1e-6f);
    for (size_t n = 0; n < n_nodes; ++n) {
        Eigen::Vector3f normalized = (full_centroids[n] - scene_min).cwiseQuotient(scene_size);
        for (size_t d = 0; d < 3; ++d) {
            normalized[d] = std::max(0.f, std::min(1.f, normalized[d]));
            position_geo.at<float>(n, d) = normalized[d];
        }
        position_geo.at<float>(n, 3) = normalized[1];
    }

    CTOCK("[Tele]0.cpu_preprocess");

    MemoryBlock2D point_feature;
    {
        CTICK("[Tele]1.obj_encoder");
        std::vector<Ort::Value> inputs;
        inputs.emplace_back(ONNX::CreateTensor(mMemoryInfo.get(), input_nodes.Get<float>(),
                                               {static_cast<int64_t>(n_nodes), 3, static_cast<int64_t>(n_pts)}));
        auto outputs = RunTeleModule("obj_encoder", inputs);
        point_feature = CopyTensor2D(outputs[0]);
        CTOCK("[Tele]1.obj_encoder");
    }

    MemoryBlock2D geo_feature;
    {
        CTICK("[Tele]2.node_geometry");
        std::vector<Ort::Value> inputs;
        inputs.emplace_back(ONNX::CreateTensor(mMemoryInfo.get(), descriptor_full.Get<float>(),
                                               {static_cast<int64_t>(n_nodes), 11}));
        inputs.emplace_back(ONNX::CreateTensor(mMemoryInfo.get(), position_geo.Get<float>(),
                                               {static_cast<int64_t>(n_nodes), 4}));
        auto outputs = RunTeleModule("node_geometry", inputs);
        geo_feature = CopyTensor2D(outputs[0]);
        CTOCK("[Tele]2.node_geometry");
    }

    std::vector<TeleYoloAssignment> yolo_assignments(n_nodes);
    if (mTeleRelationOnly && mConfigPslam->tele_yolo_object_graph) {
        for (size_t n = 0; n < n_nodes; ++n) {
            const auto &node = nodes.at(n);
            const std::string label = node->GetLabel();
            const auto iter = mLabelsName2Idx.find(label);
            if (iter == mLabelsName2Idx.end()) continue;
            yolo_assignments[n].class_idx = static_cast<int>(iter->second);
            yolo_assignments[n].score = 1.0f;
            yolo_assignments[n].overlap = 1.0f;
            {
                std::unique_lock<std::mutex> lock(node->mMutPred);
                const auto prob_iter = node->mClsProb.find(label);
                if (prob_iter != node->mClsProb.end())
                    yolo_assignments[n].score = prob_iter->second;
            }
        }
    } else if (mTeleRelationOnly && mUseTeleYolo && mTeleSessions.find("yolo") != mTeleSessions.end()) {
        cv::Mat rgb;
        Eigen::Matrix4f pose;
        CameraParameters cam{};
        {
            std::unique_lock<std::mutex> lock(mMutVisualFrame);
            if (mHasVisualFrame && !mVisualRGB.empty()) {
                rgb = mVisualRGB.clone();
                pose = mVisualPose;
                cam = mVisualCam;
            }
        }
        if (!rgb.empty() && cam.width > 0 && cam.height > 0) {
            const auto detections = RunTeleYoloSegmentation(
                    *mTeleSessions.at("yolo"),
                    mMemoryInfo.get(),
                    rgb,
                    static_cast<int>(mLabels.size()),
                    mConfigPslam->tele_yolo_conf_threshold,
                    mConfigPslam->tele_yolo_iou_threshold,
                    mConfigPslam->tele_yolo_min_mask_area);
            for (size_t n = 0; n < n_nodes; ++n) {
                yolo_assignments[n] = AssignNodeFromYolo(nodes.at(n),
                                                         detections,
                                                         pose,
                                                         cam,
                                                         mConfigPslam->tele_yolo_node_overlap_threshold);
            }
            SCLOG(VERBOSE) << "Tele YOLO produced " << detections.size()
                           << " masks and matched "
                           << std::count_if(yolo_assignments.begin(), yolo_assignments.end(),
                                            [](const TeleYoloAssignment &a) {
                                                return a.class_idx >= 0;
                                            })
                           << " / " << n_nodes << " selected nodes.";
        } else {
            SCLOG(VERBOSE) << "Tele YOLO skipped because the current visual frame is unavailable.";
        }
    }

    const size_t visual_dim = 256;
    MemoryBlock2D visual_feature;
    if (!mTeleRelationOnly)
        visual_feature = BuildTeleVisualFeature(nodes, visual_dim);

    const size_t proposal_dim = mTeleRelationOnly ? mLabels.size() + 1 : 0;
    MemoryBlock2D proposal_feature;
    if (mTeleRelationOnly) {
        proposal_feature.Resize(MemoryBlock::FLOAT, {n_nodes, proposal_dim});
        proposal_feature.Clear();
        for (size_t n = 0; n < n_nodes; ++n) {
            const auto &assignment = yolo_assignments.at(n);
            if (assignment.class_idx < 0) continue;
            const size_t cls_idx = static_cast<size_t>(assignment.class_idx);
            if (cls_idx >= mLabels.size()) continue;
            proposal_feature.at<float>(n, cls_idx) = 1.0f;
            proposal_feature.at<float>(n, mLabels.size()) = assignment.score;
        }
    }

    std::vector<char> keep_node(n_nodes, 0);
    std::vector<size_t> kept_indices;
    std::vector<float> foreground_probs(n_nodes, 0.f);
    std::vector<float> prob;
    if (mTeleRelationOnly) {
        for (size_t n = 0; n < n_nodes; ++n) {
            const auto &assignment = yolo_assignments.at(n);
            foreground_probs[n] = assignment.score;
            const bool has_yolo_label =
                    assignment.class_idx >= 0 &&
                    static_cast<size_t>(assignment.class_idx) < mLabels.size();
            const bool enough_points =
                    full_surfel_counts.at(n) >=
                    static_cast<size_t>(std::max(0, mConfigPslam->tele_yolo_min_depth_points));
            if (has_yolo_label && enough_points) {
                keep_node[n] = 1;
                kept_indices.emplace_back(n);
            }
        }
    } else {
        MemoryBlock2D foreground_logits;
        {
            std::vector<Ort::Value> inputs;
            inputs.emplace_back(ONNX::CreateTensor(mMemoryInfo.get(), point_feature.Get<float>(),
                                                   {static_cast<int64_t>(n_nodes), static_cast<int64_t>(point_feature.mDims.y)}));
            inputs.emplace_back(ONNX::CreateTensor(mMemoryInfo.get(), foreground_geo.Get<float>(),
                                                   {static_cast<int64_t>(n_nodes), 9}));
            inputs.emplace_back(ONNX::CreateTensor(mMemoryInfo.get(), visual_feature.Get<float>(),
                                                   {static_cast<int64_t>(n_nodes), static_cast<int64_t>(visual_dim)}));
            auto outputs = RunTeleModule("foreground_filter", inputs);
            foreground_logits = CopyTensor2D(outputs[0]);
        }

        for (size_t n = 0; n < n_nodes; ++n) {
            SoftmaxRow(foreground_logits.Row<float>(n), foreground_logits.mDims.y, prob);
            if (prob.size() > 1) foreground_probs[n] = prob[1];
            if (prob.size() > 1 && prob[1] >= 0.5f) {
                keep_node[n] = 1;
                kept_indices.emplace_back(n);
            }
        }
    }

    if (!debug_dir.empty()) {
        const std::string csv_path = debug_dir + "tele_foreground_debug.csv";
        const bool write_header = !FileExists(csv_path);
        std::fstream csv(csv_path, std::ios::app);
        if (write_header) {
            if (mTeleRelationOnly)
                csv << "frame,node,stable_surfels,yolo_label,yolo_score,yolo_overlap,kept\n";
            else
                csv << "frame,node,stable_surfels,foreground_prob,kept\n";
        }
        for (size_t n = 0; n < n_nodes; ++n) {
            csv << debug_frame_idx << "," << nodes.at(n)->idx << ","
                << full_surfel_counts.at(n) << ",";
            if (mTeleRelationOnly) {
                const auto &assignment = yolo_assignments.at(n);
                const std::string yolo_label =
                        assignment.class_idx >= 0 &&
                        static_cast<size_t>(assignment.class_idx) < mLabels.size() ?
                        mLabels.at(static_cast<size_t>(assignment.class_idx)) : Node::Unknown();
                csv << yolo_label << "," << assignment.score << ","
                    << assignment.overlap << ",";
            } else {
                csv << foreground_probs.at(n) << ",";
            }
            csv << int(keep_node.at(n)) << "\n";
        }
    }

    if (kept_indices.empty()) {
        SCLOG(VERBOSE) << (mTeleRelationOnly ?
                           "Tele YOLO proposals removed all selected nodes." :
                           "Tele foreground filter removed all selected nodes.");
        for (const auto &node : nodes) node->mbNeedUpdateNodeFeature = false;
        return;
    }

    const size_t n_kept = kept_indices.size();
    const size_t fuser_dim = point_feature.mDims.y + geo_feature.mDims.y +
                             (mTeleRelationOnly ? proposal_dim : visual_dim);
    MemoryBlock2D fuser_input(MemoryBlock::FLOAT, {n_nodes, fuser_dim});
    for (size_t n = 0; n < n_nodes; ++n) {
        size_t offset = 0;
        std::memcpy(fuser_input.Get<float>(n, offset), point_feature.Row<float>(n),
                    sizeof(float) * point_feature.mDims.y);
        offset += point_feature.mDims.y;
        std::memcpy(fuser_input.Get<float>(n, offset), geo_feature.Row<float>(n),
                    sizeof(float) * geo_feature.mDims.y);
        offset += geo_feature.mDims.y;
        if (mTeleRelationOnly) {
            std::memcpy(fuser_input.Get<float>(n, offset), proposal_feature.Row<float>(n),
                        sizeof(float) * proposal_dim);
        } else {
            std::memcpy(fuser_input.Get<float>(n, offset), visual_feature.Row<float>(n),
                        sizeof(float) * visual_dim);
        }
    }

    MemoryBlock2D fused_node_feature;
    {
        CTICK("[Tele]3.node_feature_fuser");
        std::vector<Ort::Value> inputs;
        inputs.emplace_back(ONNX::CreateTensor(mMemoryInfo.get(), fuser_input.Get<float>(),
                                               {static_cast<int64_t>(n_nodes), static_cast<int64_t>(fuser_dim)}));
        auto outputs = RunTeleModule("node_feature_fuser", inputs);
        fused_node_feature = CopyTensor2D(outputs[0]);
        CTOCK("[Tele]3.node_feature_fuser");
    }

    MemoryBlock2D kept_descriptor(MemoryBlock::FLOAT, {n_kept, descriptor_full.mDims.y});
    MemoryBlock2D kept_node_feature(MemoryBlock::FLOAT, {n_kept, fused_node_feature.mDims.y});
    std::vector<NodePtr> kept_nodes;
    kept_nodes.reserve(n_kept);
    for (size_t i = 0; i < n_kept; ++i) {
        const size_t old_idx = kept_indices.at(i);
        kept_nodes.emplace_back(nodes.at(old_idx));
        std::memcpy(kept_descriptor.Row<float>(i), descriptor_full.Row<float>(old_idx),
                    sizeof(float) * descriptor_full.mDims.y);
        std::memcpy(kept_node_feature.Row<float>(i), fused_node_feature.Row<float>(old_idx),
                    sizeof(float) * fused_node_feature.mDims.y);
    }

    std::vector<std::pair<size_t, size_t>> selected_edges;
    selected_edges.reserve(n_kept > 0 ? n_kept * (n_kept - 1) : 0);
    for (size_t from_local = 0; from_local < n_kept; ++from_local) {
        for (size_t to_local = 0; to_local < n_kept; ++to_local) {
            if (from_local == to_local) continue;
            selected_edges.emplace_back(from_local, to_local);
        }
    }
    SCLOG(VERBOSE) << "Tele full directed graph uses " << selected_edges.size()
                   << " edges for " << n_kept << " kept nodes.";

    MemoryBlock2D gnn_node_feature;
    MemoryBlock2D edge_feature_for_cls;
    MemoryBlock2D edge_index(MemoryBlock::INT64_T, {2, selected_edges.size()});
    if (!selected_edges.empty()) {
        for (size_t e = 0; e < selected_edges.size(); ++e) {
            edge_index.at<int64_t>(0, e) = static_cast<int64_t>(selected_edges.at(e).first);
            edge_index.at<int64_t>(1, e) = static_cast<int64_t>(selected_edges.at(e).second);
        }

        MemoryBlock2D edge_feature;
        {
            CTICK("[Tele]4.edge_encoder");
            std::vector<Ort::Value> inputs;
            inputs.emplace_back(ONNX::CreateTensor(mMemoryInfo.get(), kept_descriptor.Get<float>(),
                                                   {static_cast<int64_t>(n_kept), static_cast<int64_t>(kept_descriptor.mDims.y)}));
            inputs.emplace_back(ONNX::CreateTensor(mMemoryInfo.get(), edge_index.Get<int64_t>(),
                                                   {2, static_cast<int64_t>(selected_edges.size())}));
            auto outputs = RunTeleModule("edge_encoder", inputs);
            edge_feature = CopyTensor2D(outputs[0]);
            CTOCK("[Tele]4.edge_encoder");
        }

        {
            CTICK("[Tele]5.gnn");
            std::vector<Ort::Value> inputs;
            inputs.emplace_back(ONNX::CreateTensor(mMemoryInfo.get(), kept_node_feature.Get<float>(),
                                                   {static_cast<int64_t>(n_kept), static_cast<int64_t>(kept_node_feature.mDims.y)}));
            inputs.emplace_back(ONNX::CreateTensor(mMemoryInfo.get(), edge_feature.Get<float>(),
                                                   {static_cast<int64_t>(selected_edges.size()), static_cast<int64_t>(edge_feature.mDims.y)}));
            inputs.emplace_back(ONNX::CreateTensor(mMemoryInfo.get(), edge_index.Get<int64_t>(),
                                                   {2, static_cast<int64_t>(selected_edges.size())}));
            auto outputs = RunTeleModule("gnn", inputs);
            gnn_node_feature = CopyTensor2D(outputs[0]);
            edge_feature_for_cls = CopyTensor2D(outputs[1]);
            CTOCK("[Tele]5.gnn");
        }
    } else {
        gnn_node_feature.Resize(MemoryBlock::FLOAT, {n_kept, kept_node_feature.mDims.y});
        std::memcpy(gnn_node_feature.Get<float>(), kept_node_feature.Get<float>(),
                    sizeof(float) * kept_node_feature.size());
    }

    std::vector<char> suppress_kept_node(n_kept, 0);
    if (mTeleRelationOnly) {
        std::fstream prediction_debug_csv;
        if (!debug_dir.empty()) {
            const std::string csv_path = debug_dir + "tele_prediction_debug.csv";
            const bool write_header = !FileExists(csv_path);
            prediction_debug_csv.open(csv_path, std::ios::app);
            if (write_header)
                prediction_debug_csv << "frame,node,stable_surfels,source,label,score,overlap,suppressed\n";
        }
        for (size_t i = 0; i < n_kept; ++i) {
            const size_t original_idx = kept_indices.at(i);
            const auto &assignment = yolo_assignments.at(original_idx);
            const auto &node = kept_nodes.at(i);
            if (assignment.class_idx < 0 ||
                static_cast<size_t>(assignment.class_idx) >= mLabels.size()) {
                suppress_kept_node[i] = 1;
                node->ClearPrediction();
                continue;
            }

            std::map<std::string, float> m;
            std::map<std::string, std::pair<size_t, size_t>> mm;
            const auto fallback_size_edge = std::make_pair(node->surfels.size(), node->neighbors.size());
            const auto size_edge = sizeAndEdge.find(node->idx) != sizeAndEdge.end() ?
                                   sizeAndEdge.at(node->idx) : fallback_size_edge;
            for (const auto &label_pair : mLabels) {
                m[label_pair.second] = 0.f;
                mm[label_pair.second] = size_edge;
            }
            const std::string yolo_label = mLabels.at(static_cast<size_t>(assignment.class_idx));
            m[yolo_label] = assignment.score;
            if (prediction_debug_csv.is_open()) {
                prediction_debug_csv << debug_frame_idx << "," << node->idx << ","
                                     << full_surfel_counts.at(original_idx)
                                     << ",yolo," << yolo_label << ","
                                     << assignment.score << ","
                                     << assignment.overlap << ",0\n";
            }
            node->UpdatePrediction(m, mm, mConfigPslam->use_fusion,
                                   mConfigPslam->label_confidence_threshold);
        }
    } else {
        MemoryBlock2D node_logits;
        {
            CTICK("[Tele]6.obj_predictor");
            std::vector<Ort::Value> inputs;
            inputs.emplace_back(ONNX::CreateTensor(mMemoryInfo.get(), gnn_node_feature.Get<float>(),
                                                   {static_cast<int64_t>(n_kept), static_cast<int64_t>(gnn_node_feature.mDims.y)}));
            auto outputs = RunTeleModule("obj_predictor", inputs);
            node_logits = CopyTensor2D(outputs[0]);
            CTOCK("[Tele]6.obj_predictor");
        }

        std::fstream prediction_debug_csv;
        if (!debug_dir.empty()) {
            const std::string csv_path = debug_dir + "tele_prediction_debug.csv";
            const bool write_header = !FileExists(csv_path);
            prediction_debug_csv.open(csv_path, std::ios::app);
            if (write_header)
                prediction_debug_csv << "frame,node,stable_surfels,foreground_prob,suppressed,post_label,post_reason,tv_screen_prob,top1,top1_prob,top2,top2_prob,top3,top3_prob,top4,top4_prob,top5,top5_prob\n";
        }
        for (size_t i = 0; i < n_kept; ++i) {
            const auto &node = kept_nodes.at(i);
            SoftmaxRow(node_logits.Row<float>(i), node_logits.mDims.y, prob);
            const auto best_iter = std::max_element(prob.begin(), prob.end());
            const size_t best_idx = static_cast<size_t>(std::distance(prob.begin(), best_iter));
            const size_t original_idx = kept_indices.at(i);
            const TeleNodeLabelDecision post_decision =
                    SelectTeleNodeLabelWithPriors(prob,
                                                  mLabels,
                                                  full_bbox_mins.at(original_idx),
                                                  scene_min[1],
                                                  scene_max[1],
                                                  mConfigPslam->tele_no_prediction_threshold,
                                                  mConfigPslam->tele_prior_confidence_threshold,
                                                  mConfigPslam->tele_prior_top_k);
            const std::string post_label = TeleLabelName(mLabels, post_decision.label_idx);
            std::vector<float> adjusted_prob = prob;
            if (post_decision.force_update && post_decision.label_idx < adjusted_prob.size()) {
                const float threshold = mConfigPslam->label_confidence_threshold > 0.0f ?
                                        mConfigPslam->label_confidence_threshold : 0.30f;
                const float forced_prob = std::max(std::max(adjusted_prob.at(post_decision.label_idx),
                                                            prob.at(best_idx) + 1e-4f),
                                                   threshold + 1e-4f);
                adjusted_prob.at(post_decision.label_idx) = forced_prob;
            }
            if (prediction_debug_csv.is_open()) {
                const auto tv_iter = mLabelsName2Idx.find("tv_screen");
                const float tv_prob = (tv_iter != mLabelsName2Idx.end() && tv_iter->second < prob.size()) ?
                                      prob.at(tv_iter->second) : 0.f;
                const std::vector<size_t> order = TopKIndices(prob, 5);
                const bool sparse_suppressed =
                        ShouldSuppressTeleSparseLargeObject(post_label,
                                                            full_surfel_counts.at(original_idx),
                                                            mConfigPslam->filter_num_node);
                prediction_debug_csv << debug_frame_idx << "," << node->idx << ","
                                     << full_surfel_counts.at(original_idx) << ","
                                     << foreground_probs.at(original_idx) << ","
                                     << int(post_decision.skip_prediction || sparse_suppressed)
                                     << "," << post_label
                                     << "," << post_decision.reason
                                     << "," << tv_prob;
                for (size_t rank = 0; rank < 5; ++rank) {
                    if (rank < order.size()) {
                        const size_t cls_idx = order.at(rank);
                        prediction_debug_csv << ","
                                             << (cls_idx < mLabels.size() ? mLabels.at(cls_idx) : Node::Unknown())
                                             << "," << prob.at(cls_idx);
                    } else {
                        prediction_debug_csv << ",,";
                    }
                }
                prediction_debug_csv << "\n";
            }
            if (post_decision.skip_prediction) {
                suppress_kept_node[i] = 1;
                node->ClearPrediction();
                SCLOG(VERBOSE) << "Skip Tele node " << node->idx
                               << " prediction because top-1 confidence is below "
                               << mConfigPslam->tele_no_prediction_threshold << ".";
                continue;
            }
            if (ShouldSuppressTeleSparseLargeObject(post_label, full_surfel_counts.at(original_idx),
                                                    mConfigPslam->filter_num_node)) {
                suppress_kept_node[i] = 1;
                node->ClearPrediction();
                SCLOG(VERBOSE) << "Suppress sparse Tele node " << node->idx
                               << " predicted as " << post_label
                               << " with " << full_surfel_counts.at(original_idx)
                               << " stable surfels.";
                continue;
            }

            std::map<std::string, float> m;
            std::map<std::string, std::pair<size_t, size_t>> mm;
            const auto fallback_size_edge = std::make_pair(node->surfels.size(), node->neighbors.size());
            const auto size_edge = sizeAndEdge.find(node->idx) != sizeAndEdge.end() ?
                                   sizeAndEdge.at(node->idx) : fallback_size_edge;
            const size_t n_cls = std::min(adjusted_prob.size(), mLabels.size());
            for (size_t j = 0; j < n_cls; ++j) {
                m[mLabels.at(j)] = adjusted_prob.at(j);
                mm[mLabels.at(j)] = size_edge;
            }
            node->UpdatePrediction(m, mm, mConfigPslam->use_fusion,
                                   mConfigPslam->label_confidence_threshold);
        }
    }

    if (!selected_edges.empty()) {
        MemoryBlock2D edge_logits;
        {
            CTICK("[Tele]7.rel_predictor");
            std::vector<Ort::Value> inputs;
            inputs.emplace_back(ONNX::CreateTensor(mMemoryInfo.get(), edge_feature_for_cls.Get<float>(),
                                                   {static_cast<int64_t>(selected_edges.size()), static_cast<int64_t>(edge_feature_for_cls.mDims.y)}));
            auto outputs = RunTeleModule("rel_predictor", inputs);
            edge_logits = CopyTensor2D(outputs[0]);
            CTOCK("[Tele]7.rel_predictor");
        }

        const size_t n_rel = std::min(edge_logits.mDims.y, mRelationships.size());
        for (size_t e = 0; e < selected_edges.size(); ++e) {
            if (suppress_kept_node.at(selected_edges.at(e).first) ||
                suppress_kept_node.at(selected_edges.at(e).second))
                continue;
            EdgePtr edge = std::make_shared<PSLAM::Edge>();
            edge->nodeFrom = kept_nodes.at(selected_edges.at(e).first)->idx;
            edge->nodeTo = kept_nodes.at(selected_edges.at(e).second)->idx;
            std::map<std::string, float> prop;
            auto *data = edge_logits.Row<float>(e);
            for (size_t j = 0; j < n_rel; ++j)
                prop[mRelationships.at(j)] = Sigmoid(data[j]);
            edge->UpdatePrediction(prop, mConfigPslam->use_fusion);
            mpGraph->AddEdge(edge);
        }
    }

    for (const auto &node : nodes) node->mbNeedUpdateNodeFeature = false;
}

void GraphPredictor::UpdatePrediction(const std::map<int,NodePtr> &vNodes,
                                      const std::map<int,std::pair<size_t,size_t>> &sizeAndEdge) {
    CTICK_RESET("[GraphPredictor::UpdatePrediction]1.lock_updateNode");
    CTICK_RESET("[GraphPredictor::UpdatePrediction]2.lock_select_edge");
    CTICK_RESET("[GraphPredictor::UpdatePrediction]3.lock_update_edge");
    CTICK_RESET("[GraphPredictor::UpdatePrediction]4.lock_update_instances");
    SCLOG(VERBOSE) << "Prediction Updated start.";
    const auto &nodes = mpGraph->nodes;
    auto last_level = std::to_string(size_t(GetParams().at("n_layers").int_value()));

    /// Collect the neighbor of the selected nodes

    std::map<size_t, size_t> idx2seg;
    std::map<size_t, size_t> seg2idx;
    /// Filter out nodes without computed feature
    std::vector<NodePtr> selected_nodes;
    std::vector<std::pair<size_t,size_t>> filtered_sizeAndEdge;
    SCLOG(VERBOSE) << "filter input nodes.";
    {
        for(const auto &pair:vNodes) {
            const auto &node = pair.second;
            if(node->mFeatures.find(last_level) != node->mFeatures.end()) {
                selected_nodes.push_back(node);
                filtered_sizeAndEdge.push_back(sizeAndEdge.at(pair.first));
            }
        }
    }
    size_t n_nodes = selected_nodes.size();

    std::map<int,std::set<int>> has_same_part_nodes;
    if(1) /// Node
    {
        int64_t dim_obj_f = GetParams().at("dim_o_f").int_value();
        if (!selected_nodes.empty()) {
            MemoryBlock2D nodeFeatures(MemoryBlock::FLOAT,
                                       {selected_nodes.size(), static_cast<unsigned long>(dim_obj_f)});
            SCLOG(VERBOSE) << "collect node features.";
            for (size_t i = 0; i < n_nodes; ++i) {
                const auto &last_node_feature = selected_nodes.at(i)->mFeatures.at(last_level)->Get<float>();
                memcpy(nodeFeatures.Row<float>(i), last_node_feature, sizeof(float) * dim_obj_f);
            }

            if (mDebug) ONNX::PrintVector("nodeFeatures", nodeFeatures.Get<float>(), std::vector<size_t>{nodeFeatures.mDims.x, nodeFeatures.mDims.y});

            SCLOG(VERBOSE) << "predict.";
            std::vector<Ort::Value> node_features;
            node_features.push_back(ONNX::CreateTensor(mMemoryInfo.get(), nodeFeatures.Get<float>(),
                                                       {static_cast<long>(n_nodes),
                                                        static_cast<long>(dim_obj_f)}));
            auto objcls_prob = mEatGCN->Compute(PSLAM::EatGCN::OP_CLS_OBJ, node_features);
            size_t num_cls = objcls_prob[0].GetTensorTypeAndShapeInfo().GetShape()[1];
            assert(mLabels.size() == num_cls);
            SCLOG(VERBOSE) << "write back.";
            for (size_t i = 0; i < n_nodes; ++i) {
                const auto &node = selected_nodes.at(i);
                std::map<std::string, float> m;
                std::map<std::string, std::pair<size_t, size_t>> mm;
                float *data = objcls_prob[0].GetTensorMutableData<float>() + i * num_cls;
                for (size_t j = 0; j < num_cls; ++j) {
                    m[mLabels.at(j)] = expf(data[j]); // logsoftmax -> softmax
                    mm[mLabels.at(j)] = filtered_sizeAndEdge.at(i);
                }
                node->UpdatePrediction(m, mm, mConfigPslam->use_fusion,
                                       mConfigPslam->label_confidence_threshold);

                std::set<int> checked_nodes;
                auto has_same_nodes = has_same_part_nodes[node->idx];
                CTICK("[GraphPredictor::UpdatePrediction]1.lock_updateNode");
                {
                    std::unique_lock<std::mutex> lock(mpGraph->mMutNode);
                    find_nodes_has_same_part(nodes, node->idx, node->idx, has_same_nodes, checked_nodes);
                }
                CTOCK("[GraphPredictor::UpdatePrediction]1.lock_updateNode");
            }
        }
    }
    if(1) /// Edge
    {
        const auto &edges = mpGraph->edges;
        std::vector<EdgePtr> selected_edges;
        int64_t dim_rel_f = GetParams().at("dim_r_f").int_value();
        if(n_nodes==0){
            SCLOG(VERBOSE) << "got 0 nodes. return";
            return;
        }
        for(size_t i=0;i<n_nodes;++i) {
            idx2seg[i] = selected_nodes.at(i)->idx;
            seg2idx[selected_nodes.at(i)->idx] = i;
        }


        for (auto &selected_node : selected_nodes) {
            const auto &n = selected_node->idx;
            if (seg2idx.find(n) == seg2idx.end()) continue;
            CTICK("[GraphPredictor::UpdatePrediction]2.lock_select_edge");
            std::unique_lock<std::mutex> lock(mpGraph->mMutEdge);
            for (const auto &nn : selected_node->neighbors) {
                if (vNodes.find(nn) == vNodes.end()) continue;
                if (seg2idx.find(nn) == seg2idx.end()) continue;
                if (seg2idx.at(n) == seg2idx.at(nn)) continue;
                if (edges.find({n, nn}) == edges.end()) continue;
                if (edges.at({n, nn})->mFeatures.find(last_level) ==
                    edges.at({n, nn})->mFeatures.end())
                    continue;
                selected_edges.push_back(edges.at({n, nn}));
            }
            CTOCK("[GraphPredictor::UpdatePrediction]2.lock_select_edge");
        }

        MemoryBlock2D edgeFeatures(MemoryBlock::FLOAT,
                                   {selected_edges.size(), static_cast<unsigned long>(dim_rel_f)});
        for (size_t i = 0; i < selected_edges.size(); ++i) {
            const auto &last_node_feature = selected_edges.at(i)->mFeatures.at(last_level)->Get<float>();
            memcpy(edgeFeatures.Row<float>(i), last_node_feature, sizeof(float) * dim_rel_f);
        }

        if (mDebug) ONNX::PrintVector("edgeFeatures", edgeFeatures.Get<float>(), std::vector<size_t>{edgeFeatures.mDims.x, edgeFeatures.mDims.y});

        std::vector<Ort::Value> edge_features;
        edge_features.push_back(ONNX::CreateTensor(mMemoryInfo.get(), edgeFeatures.Get<float>(),
                                                   {static_cast<long>(selected_edges.size()),
                                                    static_cast<long>(dim_rel_f)}));
        auto relcls_prob = mEatGCN->Compute(PSLAM::EatGCN::OP_CLS_REL, edge_features);

        size_t num_rel = relcls_prob[0].GetTensorTypeAndShapeInfo().GetShape()[1];
        assert(num_rel == mRelationships.size());
        for(size_t i=0;i<selected_edges.size();++i){
            auto &edge = selected_edges.at(i);
            float *data = relcls_prob[0].GetTensorMutableData<float>() + i * num_rel;
            std::map<std::string, float> prop;
            for(size_t j=0;j<num_rel;++j) {
                prop[mRelationships.at(j)] = GetParams().at("multi_rel_outputs").bool_value()? data[j] : expf(data[j]); // logsoftmax->softmax
            }
            CTICK("[GraphPredictor::UpdatePrediction]3.lock_update_edge");
            edge->UpdatePrediction(prop,mConfigPslam->use_fusion);
            CTOCK("[GraphPredictor::UpdatePrediction]3.lock_update_edge");
        }
    }

    /// Update Instances
    if(1)
    {
        int idx_same_part = -1;
        for (auto &pair:mRelationships) {
            if (pair.second == Edge::Same()) {
                idx_same_part = int(pair.first);
                break;
            }
        }
        assert(idx_same_part >= 0);
        std::set<int> checked_nodes;
        CTICK("[GraphPredictor::UpdatePrediction]4.lock_update_instances");
        for (const auto &pair:selected_nodes) {
            if(has_same_part_nodes.find(pair->idx) == has_same_part_nodes.end()) continue;
            const auto& pre_same_nodes = has_same_part_nodes.at(pair->idx);
            std::set<int> found_nodes;
            std::unique_lock<std::mutex> lock(mpGraph->mMutNode);
            find_nodes_has_same_part(nodes, pair->idx, idx_same_part, found_nodes, checked_nodes);


            // check if there are nodes previously has the same part but removed later
            std::set<int> should_seperate;
            for(auto i : pre_same_nodes)
                if(found_nodes.find(i) == found_nodes.end()) should_seperate.insert(i);
            for(auto i : should_seperate) {
                nodes.at(i)->instance_idx = i;
            }


            if(found_nodes.empty()) continue;
            // find the smallest idx as their instance id
            found_nodes.insert(pair->idx);// add itself
            int instance_id = int(pair->idx);
            nodes.at(pair->idx)->instance_idx = instance_id; // set to itself, in case "same part" relationship is removed.
            for(auto& i : found_nodes)
                if(i < instance_id) instance_id = i;
            for(auto& i : found_nodes)
                nodes.at(i)->instance_idx = instance_id;


            SCLOG(DEBUG) << pair->idx << ": There are  " << found_nodes.size() << " have same part relationship with node " << pair->idx << ". The instance id is: " << instance_id;
            std::stringstream ss;
            for (auto s : found_nodes) {
                ss << s << " ";
            }
            SCLOG(DEBUG) << ss.str();
        }
        CTOCK("[GraphPredictor::UpdatePrediction]4.lock_update_instances");
    }
    SCLOG(VERBOSE) << "Prediction Updated end";
}

bool GraphPredictor::Pin() {
    if(mThread.valid()) {
        //wait until stop
        if( mThread.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
            mThread.get();
            return true;
        }
    }
    return false;
}

void GraphPredictor::SetThread(bool option) {mbThread = option; }

bool GraphPredictor::Updated(){return mUpdated;}
void GraphPredictor::SetUpdate(bool option){mUpdated=option;}
const std::map<std::string, json11::Json>& GraphPredictor::GetParams(){
    if (mUseTeleModules) return mTeleParams;
    return mEatGCN->mParams;
}

std::map<std::string,std::vector<std::pair<int,float>>> GraphPredictor::GetTimes() {
    return mTimes;
}
#endif
