#define STB_IMAGE_IMPLEMENTATION
#ifdef __APPLE__
/* Defined before OpenGL and GLUT includes to avoid deprecation messages */
#define GL_SILENCE_DEPRECATION
#endif
#include <iostream>
#include <fstream>
#include <chrono>
#include <algorithm>
#include <iomanip>
#include <dataLoader/dataset_loader_facotry.h>
#include "../../Utils/parser.hpp"
#include <ORUtils/Logging.h>
#include <ORUtils/PathTool.hpp>

#ifdef COMPILE_WITH_PSLAM_GUI
#include "../../libGraphSLAMGUI/GraphSLAMGUI.h"
#include "../../Utils/EvaluationHelper.h"
#else
#include "../../libGraphSLAM/GraphSLAM.h"
#include "../Utils/EvaluationHelper.h"
#include "../../Utils/SceneGraphJsonWriter.h"
#ifdef COMPILE_WITH_ASSIMP
#include "../libGUI3D/libGUI3D/GUI3D.h"
#include "../../renderer/RendererFactory.h"
#endif
#endif
#ifdef COMPILE_WITH_ROS_PUBLISH
#include <rclcpp/rclcpp.hpp>
#include "../../libGraphSLAM/rosPublisher/SceneGraphRosPublisher.h"
#endif

struct Params{
    std::string pth_in;
    std::string pth_out = "./";
    std::string pth_model;
    std::string scene_graph_json_dir;
    std::string scene_graph_ros_topic;
    std::string save_name = "inseg.ply";
    int min_pyr_level=2;

    float depth_edge_threshold = -1; // -1: use default.

    /// Use rendered view from a given mesh (for ScanNet)
    bool use_render=true;
    bool save=true;
    bool save_graph=true;
    bool save_graph_ply = true;
    bool save_surfel_ply = true;
    bool save_time = true;

    /// Predict semantic scene graph
    bool use_predict=true;
    bool full_prop=true;
    bool binary=false;
    bool fusion=true;
    ///
    bool thread=true;

    bool verbose=false;
    int segment_filter=512;
    float label_confidence_threshold = -1.0f;
    int tele_network_geometry_compatibility = -1;
    float tele_no_prediction_threshold = -1.0f;
    float tele_prior_confidence_threshold = -1.0f;
    int tele_prior_top_k = -1;
    std::string tele_debug_dir;
    std::string tele_yolo_model;
    float tele_yolo_conf_threshold = -1.0f;
    float tele_yolo_iou_threshold = -1.0f;
    int tele_yolo_min_mask_area = -1;
    int tele_yolo_min_depth_points = -1;
    float tele_yolo_node_overlap_threshold = -1.0f;
    int tele_yolo_object_graph = -1;
    float tele_yolo_track_assoc_distance = -1.0f;
    int tele_yolo_track_max_age = -1;
    int tele_yolo_object_min_points = -1;
    int tele_yolo_object_sample_stride = -1;
    int tele_yolo_object_max_points_per_mask = -1;
    int tele_yolo_track_max_points = -1;
    float tele_yolo_object_voxel_size = -1.0f;
};

static bool FileExists(const std::string &path) {
    std::ifstream file(path, std::ios::in);
    return file.good();
}

static std::string WithTrailingSlash(std::string path) {
    if (!path.empty() && path.back() != '/') path += "/";
    return path;
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

static bool IsTeleRelationOnlyOnnxModuleDir(const std::string &path) {
    const auto folder = ResolveTeleOnnxModuleDir(path);
    return IsTeleOnnxModuleDir(path) &&
           (!FileExists(folder + "obj_predictor.onnx") ||
            !FileExists(folder + "foreground_filter.onnx"));
}

static std::string DefaultTeleYoloModelPath() {
    return "/home/gao/codes/TeleScenegraphFusion-visual-feature/out/yolo/"
           "yolo11n_seg_teledataset-3/weights/best.onnx";
}

static void ApplyTeleInSegConfig(PSLAM::ConfigPSLAM &config) {
    config.inseg_config.remove_dynamic_point = false;
    config.inseg_config.surfel_invalid_frame_threshold = 30;
    config.inseg_config.surfel_stable_frame_threshold = 2;
    config.inseg_config.update_point_angle_threshold = 75.0f;
    config.inseg_config.update_point_dot_product_threshold = 30.0f;
    config.inseg_config.label_merge_confidence_threshold = 3;
    config.inseg_config.label_min_size = 50;
    config.inseg_config.label_point_dot_product_threshold = 8.0f;
    config.inseg_config.depth_edge_threshold = 0.95f;
    config.inseg_config.label_size_threshold = 80;
    config.inseg_config.label_merge_overlap_ratio = 0.45f;
    config.inseg_config.label_region_ratio = 0.45f;

    config.main_config.uncertainty_coefficient_depth = 0.0000285;
    config.main_config.min_pyr_level = 2;
    config.main_config.max_pyr_level = 6;

    config.map_config.near_depth_threshold = 300.0;
    config.map_config.far_depth_threshold = 10000.0;
    config.map_config.num_iterations_bilateral_filtering = 2;

    config.segmentation_config.min_region_size = 10;
}

void ParseCommondLine(int argc, char **argv, Params &params) {
    tools::Parser parser(argc,argv);
    parser.addOption(pkgcname("pth_in", &params.pth_in), "", true);
    parser.addOption(pkgcname("pth_out", &params.pth_out),"");
    parser.addOption(pkgcname("pth_model", &params.pth_model),"path to the folder contains models",false);

    parser.addOption(pkgcname("save_name", &params.save_name),"");

    parser.addOption(pkgcname("min_pyr_level", &params.min_pyr_level),
                     "which level to run InSeg");
    parser.addOption(pkgcname("segment_filter", &params.segment_filter),
                     "Filter out segment that has not enough surfels.");
    parser.addOption(pkgcname("depth_edge_threshold", &params.depth_edge_threshold),
                     "depth_edge_threshold for InSeg");

    parser.addOption(pkgcname("rendered", &params.use_render), "use rendered depth");
    parser.addOption(pkgcname("full_prop", &params.full_prop), "return full prop or just the maximum.");
    parser.addOption(pkgcname("fusion", &params.fusion), "use fusion.");
    parser.addOption(pkgcname("binary", &params.binary), "save ply to binary.");
    parser.addOption(pkgcname("thread", &params.thread), "use thread on scene graph prediction.");
    parser.addOption(pkgcname("verbose", &params.verbose), "verbose.");
    parser.addOption(pkgcname("prediction", &params.use_predict), "do graph prediction.");
    parser.addOption(pkgcname("label_confidence_threshold", &params.label_confidence_threshold),
                     "minimum node label confidence. <0 uses dataset default; 0 disables it.");
    parser.addOption(pkgcname("tele_network_geometry_compatibility",
                              &params.tele_network_geometry_compatibility),
                     "Tele ONNX only: 1 uses legacy pre-quaternion-fix network geometry; 0 uses reconstruction geometry; <0 uses default.");
    parser.addOption(pkgcname("tele_no_prediction_threshold",
                              &params.tele_no_prediction_threshold),
                     "Tele ONNX only: skip node prediction when top-1 class probability is below this value; <0 uses default 0.10.");
    parser.addOption(pkgcname("tele_prior_confidence_threshold",
                              &params.tele_prior_confidence_threshold),
                     "Tele ONNX only: use location/category priors when top-1 class probability is below this value; <0 follows label_confidence_threshold/default 0.30.");
    parser.addOption(pkgcname("tele_prior_top_k",
                              &params.tele_prior_top_k),
                     "Tele ONNX only: number of top class candidates checked by location/category priors; <0 uses default 3.");
    parser.addOption(pkgcname("tele_debug_dir", &params.tele_debug_dir),
                     "Optional Tele ONNX debug dump folder.");
    parser.addOption(pkgcname("tele_yolo_model", &params.tele_yolo_model),
                     "Tele relation-only: YOLO segmentation ONNX path.");
    parser.addOption(pkgcname("tele_yolo_conf_threshold",
                              &params.tele_yolo_conf_threshold),
                     "Tele relation-only: YOLO confidence threshold; <0 uses default 0.35.");
    parser.addOption(pkgcname("tele_yolo_iou_threshold",
                              &params.tele_yolo_iou_threshold),
                     "Tele relation-only: YOLO NMS IoU threshold; <0 uses default 0.70.");
    parser.addOption(pkgcname("tele_yolo_min_mask_area",
                              &params.tele_yolo_min_mask_area),
                     "Tele relation-only: minimum YOLO mask area in pixels; <0 uses default 64.");
    parser.addOption(pkgcname("tele_yolo_min_depth_points",
                              &params.tele_yolo_min_depth_points),
                     "Tele relation-only: minimum valid depth points in a YOLO mask; <0 uses default 50.");
    parser.addOption(pkgcname("tele_yolo_node_overlap_threshold",
                              &params.tele_yolo_node_overlap_threshold),
                     "Tele relation-only: minimum projected node/mask overlap ratio; <0 uses default 0.05.");
    parser.addOption(pkgcname("tele_yolo_object_graph",
                              &params.tele_yolo_object_graph),
                     "Tele relation-only: 1 uses YOLO mask/depth object tracks as graph nodes; 0 keeps InSeg nodes; <0 enables it for relation-only models.");
    parser.addOption(pkgcname("tele_yolo_track_assoc_distance",
                              &params.tele_yolo_track_assoc_distance),
                     "Tele YOLO object graph: max same-class track association distance in meters; <0 uses default 0.60.");
    parser.addOption(pkgcname("tele_yolo_track_max_age",
                              &params.tele_yolo_track_max_age),
                     "Tele YOLO object graph: remove tracks unseen for this many frames; <0 keeps previously seen objects.");
    parser.addOption(pkgcname("tele_yolo_object_min_points",
                              &params.tele_yolo_object_min_points),
                     "Tele YOLO object graph: minimum accumulated points per object node; <0 uses default 50.");
    parser.addOption(pkgcname("tele_yolo_object_sample_stride",
                              &params.tele_yolo_object_sample_stride),
                     "Tele YOLO object graph: legacy pixel stride for mask/depth backprojection; raw-map mode ignores it.");
    parser.addOption(pkgcname("tele_yolo_object_max_points_per_mask",
                              &params.tele_yolo_object_max_points_per_mask),
                     "Tele YOLO object graph: max raw-map surfels added per mask per frame; <0 uses default 50000.");
    parser.addOption(pkgcname("tele_yolo_track_max_points",
                              &params.tele_yolo_track_max_points),
                     "Tele YOLO object graph: max accumulated raw-map surfels per track; <0 uses default 200000.");
    parser.addOption(pkgcname("tele_yolo_object_voxel_size",
                              &params.tele_yolo_object_voxel_size),
                     "Tele YOLO object graph: legacy voxel size in meters; raw-map mode keeps surfel ids and defaults to 0.005.");

    parser.addOption(pkgcname("scene_graph_json_dir", &params.scene_graph_json_dir),
                     "If set, write the 3D scene graph to this directory as one JSON file per frame.");
    parser.addOption(pkgcname("scene_graph_ros_topic", &params.scene_graph_ros_topic),
                     "If set, publish per-frame scene graph changes as a scene_msgs/TeleScene ROS2 topic "
                     "with this name (only added/changed/removed nodes and edges are included). "
                     "Requires the project to be built with -DBUILD_ROS_PUBLISH=ON.");
    parser.addOption(pkgcname("save", &params.save), "");
    parser.addOption(pkgcname("save_graph", &params.save_graph), "");
    parser.addOption(pkgcname("save_graph_ply", &params.save_graph_ply), "output graph as ply (with semantic id)");
    parser.addOption(pkgcname("save_surfel_ply", &params.save_surfel_ply), "output surfels to ply (inseg output)");
    parser.addOption(pkgcname("save_time", &params.save_time), "");

    auto status = parser.showMsg(params.verbose);
    if(status<1)
        exit(-1);
    if(status==2)
        exit(0);
}

#ifndef COMPILE_WITH_PSLAM_GUI
#ifdef COMPILE_WITH_ASSIMP
class Renderer : SC::GUI3D {
public:
    Renderer(int width, int height, const std::string &path, bool align):SC::GUI3D("render",width,height,false){
        std::string folder, scan_id;
        PSLAM::MeshRenderType type;
        if(path.find("scene") != std::string::npos) {
            auto parent_folder = tools::PathTool::find_parent_folder(path, 1);
            scan_id = tools::PathTool::getFileName(parent_folder);
            folder =  tools::PathTool::find_parent_folder(parent_folder, 1);
            type = PSLAM::MeshRenderType_ScanNet;
        } else {
            auto seq_folder = tools::PathTool::find_parent_folder(path,1);
            scan_id = tools::PathTool::getFileName(seq_folder);
            folder = tools::PathTool::find_parent_folder(seq_folder,1);
            type = PSLAM::MeshRenderType_3RScan;
        }
        mMeshRenderer.reset( PSLAM::MakeMeshRenderer(width, height, folder,scan_id,type,align) );
    }

    void Render(cv::Mat &rgb, cv::Mat &depth, const Eigen::Matrix4f &pose, const PSLAM::CameraParameters &params){
        auto view_pose = glUtil::GetViewMatrix(pose);
        auto proj = glUtil::Perspective<float>(params.fx,params.fy,
                                               params.cx,params.cy,
                                               params.width,params.height,
                                               glCam->projection_control_->near,glCam->projection_control_->far);
        mMeshRenderer->Render(proj,view_pose,glCam->projection_control_->near,glCam->projection_control_->far);
        depth = mMeshRenderer->GetDepth();
    }
private:
    std::unique_ptr<PSLAM::MeshRendererInterface> mMeshRenderer;
};
#else
class Renderer{
public:

    Renderer(int width, int height, const std::string &path){
        throw std::runtime_error("did not compile with assimp");
    }

    void Render(cv::Mat &rgb, cv::Mat &depth, const Eigen::Matrix4f &pose, const PSLAM::CameraParameters &params){
        throw std::runtime_error("did not compile with assimp");
    }
};
#endif
#endif

int main(int argc, char** argv) {
    Params params;
    ParseCommondLine(argc, argv, params);
#ifdef COMPILE_WITH_ROS_PUBLISH
    if (!params.scene_graph_ros_topic.empty()) rclcpp::init(argc, argv);
#endif
#ifdef NDEBUG
    if(params.verbose) SCLOG_ON(VERBOSE);
    else SCLOG_ON(INFO);
#else
    SCLOG_ON(VERBOSE);
#endif

    if (params.save) tools::PathTool::create_folder(params.pth_out);
    auto path = params.pth_in;
    if (path.find(".txt") != std::string::npos) {
        std::ifstream file(path);
        assert(file.is_open());
        std::getline(file, path, '\n');
    }
    SCLOG(INFO) << "Buliding data loader...";
    std::shared_ptr<PSLAM::DatasetLoader_base> dataset_loader_;
    dataset_loader_.reset(PSLAM::DataLoaderFactory::Make(path));
    dataset_loader_->Reset();

    const bool isTeleDataset = dataset_loader_->GetBase()->datasetType == PSLAM::DATASET_TELE;
    if (isTeleDataset && params.use_render) {
        SCLOG(WARNING) << "Tele dataset does not provide a mesh renderer; disabling rendered depth.";
        params.use_render = false;
    }

    if(params.use_render) {
        if (path.find("scene") == std::string::npos) dataset_loader_->GetCamParamDepth() = dataset_loader_->GetCamParamRGB();
    }

#ifndef COMPILE_WITH_PSLAM_GUI
    std::unique_ptr<Renderer> renderer;
    if(params.use_render) {
        SCLOG(INFO) << "Building renderer...";
        renderer = std::make_unique<Renderer>(dataset_loader_->GetCamParamDepth().width,
                                              dataset_loader_->GetCamParamDepth().height,
                                              path, true
        );
    }
#endif

    SCLOG(INFO) << "Building framework...";
    PSLAM::ConfigPSLAM configPslam;
    if (isTeleDataset) {
        ApplyTeleInSegConfig(configPslam);
        SCLOG(INFO) << "Tele dataset detected; using InSeg parameters from py_inseg_config.";
    }
    /// Adjust configurations according to dataset and level
    if(params.pth_model.empty())
        configPslam.graph_predict = false;
    else
    {
        configPslam.pth_model = isTeleDataset ?
                                ResolveTeleOnnxModuleDir(params.pth_model) :
                                params.pth_model;
        if (isTeleDataset)
            configPslam.pth_dataset = tools::PathTool::find_parent_folder(path, 1);
        configPslam.use_fusion = params.fusion;
        configPslam.use_thread = params.thread;
        configPslam.filter_num_node = params.segment_filter;
        configPslam.tele_debug_dir = params.tele_debug_dir;
        if (!configPslam.tele_debug_dir.empty())
            tools::PathTool::create_folder(configPslam.tele_debug_dir);
        if (params.label_confidence_threshold >= 0.0f)
            configPslam.label_confidence_threshold = params.label_confidence_threshold;
        configPslam.graph_predict=params.use_predict;
        if (!isTeleDataset)
            configPslam.main_config.max_pyr_level = 4;
        configPslam.main_config.min_pyr_level = params.min_pyr_level;

        if (isTeleDataset && IsTeleOnnxModuleDir(configPslam.pth_model)) {
            configPslam.n_pts = 1024;
            if (params.label_confidence_threshold < 0.0f)
                configPslam.label_confidence_threshold = 0.30f;
            if (params.tele_no_prediction_threshold >= 0.0f)
                configPslam.tele_no_prediction_threshold = params.tele_no_prediction_threshold;
            if (params.tele_prior_confidence_threshold >= 0.0f)
                configPslam.tele_prior_confidence_threshold = params.tele_prior_confidence_threshold;
            else if (configPslam.label_confidence_threshold > 0.0f)
                configPslam.tele_prior_confidence_threshold = configPslam.label_confidence_threshold;
            if (params.tele_prior_top_k > 0)
                configPslam.tele_prior_top_k = params.tele_prior_top_k;
            configPslam.tele_network_geometry_compatibility =
                    params.tele_network_geometry_compatibility > 0;
            if (!params.tele_yolo_model.empty()) {
                configPslam.tele_yolo_model_path = params.tele_yolo_model;
            } else if (IsTeleRelationOnlyOnnxModuleDir(configPslam.pth_model) &&
                       FileExists(DefaultTeleYoloModelPath())) {
                configPslam.tele_yolo_model_path = DefaultTeleYoloModelPath();
            }
            if (params.tele_yolo_conf_threshold >= 0.0f)
                configPslam.tele_yolo_conf_threshold = params.tele_yolo_conf_threshold;
            if (params.tele_yolo_iou_threshold >= 0.0f)
                configPslam.tele_yolo_iou_threshold = params.tele_yolo_iou_threshold;
            if (params.tele_yolo_min_mask_area >= 0)
                configPslam.tele_yolo_min_mask_area = params.tele_yolo_min_mask_area;
            if (params.tele_yolo_min_depth_points >= 0)
                configPslam.tele_yolo_min_depth_points = params.tele_yolo_min_depth_points;
            if (params.tele_yolo_node_overlap_threshold >= 0.0f)
                configPslam.tele_yolo_node_overlap_threshold =
                        params.tele_yolo_node_overlap_threshold;
            if (params.tele_yolo_track_assoc_distance >= 0.0f)
                configPslam.tele_yolo_track_assoc_distance =
                        params.tele_yolo_track_assoc_distance;
            if (params.tele_yolo_track_max_age >= 0)
                configPslam.tele_yolo_track_max_age = params.tele_yolo_track_max_age;
            if (params.tele_yolo_object_min_points >= 0)
                configPslam.tele_yolo_object_min_points = params.tele_yolo_object_min_points;
            if (params.tele_yolo_object_sample_stride >= 0)
                configPslam.tele_yolo_object_sample_stride =
                        std::max(1, params.tele_yolo_object_sample_stride);
            if (params.tele_yolo_object_max_points_per_mask >= 0)
                configPslam.tele_yolo_object_max_points_per_mask =
                        params.tele_yolo_object_max_points_per_mask;
            if (params.tele_yolo_track_max_points >= 0)
                configPslam.tele_yolo_track_max_points = params.tele_yolo_track_max_points;
            if (params.tele_yolo_object_voxel_size > 0.0f)
                configPslam.tele_yolo_object_voxel_size = params.tele_yolo_object_voxel_size;
            if (params.segment_filter == 512) {
                params.segment_filter = 10;
                configPslam.filter_num_node = 10;
                SCLOG(INFO) << "Tele ONNX model detected; using segment_filter=10 and n_pts=1024.";
            } else {
                SCLOG(INFO) << "Tele ONNX model detected; using n_pts=1024.";
            }
            if (IsTeleRelationOnlyOnnxModuleDir(configPslam.pth_model)) {
                configPslam.tele_yolo_object_graph =
                        params.tele_yolo_object_graph < 0 ||
                        params.tele_yolo_object_graph > 0;
                SCLOG(INFO) << "Tele relation-only ONNX model detected; node labels come from YOLO proposals.";
                if (!configPslam.tele_yolo_model_path.empty()) {
                    SCLOG(INFO) << "Using Tele YOLO segmentation ONNX: "
                                << configPslam.tele_yolo_model_path;
                } else {
                    SCLOG(WARNING) << "Tele relation-only model has no YOLO ONNX path; "
                                   << "relations will use existing node labels only.";
                }
                if (configPslam.tele_yolo_object_graph)
                    SCLOG(INFO) << "Tele YOLO object graph enabled: graph nodes come from "
                                << "YOLO mask/depth tracks.";
            } else if (params.tele_yolo_object_graph >= 0) {
                configPslam.tele_yolo_object_graph =
                        params.tele_yolo_object_graph > 0;
            }
        }
        if (configPslam.tele_network_geometry_compatibility)
            SCLOG(INFO) << "Tele legacy network geometry compatibility enabled.";
        if (configPslam.label_confidence_threshold > 0.0f)
            SCLOG(INFO) << "Using node label confidence threshold: "
                        << configPslam.label_confidence_threshold;
        if (isTeleDataset && IsTeleOnnxModuleDir(configPslam.pth_model))
            SCLOG(INFO) << "Using Tele node postprocess thresholds: no_prediction="
                        << configPslam.tele_no_prediction_threshold
                        << ", prior_confidence="
                        << configPslam.tele_prior_confidence_threshold
                        << ", prior_top_k="
                        << configPslam.tele_prior_top_k;
    }

    if(params.depth_edge_threshold == -1) {
        SCLOG(VERBOSE) << "use predefined depth edge threshold: " << configPslam.inseg_config.depth_edge_threshold;
        if (!isTeleDataset) {
            if (configPslam.main_config.min_pyr_level == 1)
                configPslam.inseg_config.depth_edge_threshold = 0.99f;
            else if (configPslam.main_config.min_pyr_level == 2)
                configPslam.inseg_config.depth_edge_threshold = 0.99f;
            else if (configPslam.main_config.min_pyr_level == 3)
                configPslam.inseg_config.depth_edge_threshold = 0.98f;
        }
    } else {
        SCLOG(VERBOSE) << "use custom depth edge threshold: " << params.depth_edge_threshold;
        configPslam.inseg_config.depth_edge_threshold = params.depth_edge_threshold;
    }

    PSLAM::GraphSLAM graphSlam(&configPslam, dataset_loader_->GetCamParamDepth());

    using Clock = std::chrono::steady_clock;
#ifndef COMPILE_WITH_PSLAM_GUI
    Clock::time_point end_to_end_start;
    Clock::time_point end_to_end_end;
    size_t processed_frame_count = 0;
    double end_to_end_wall_ms = 0.0;
#endif

#ifdef COMPILE_WITH_PSLAM_GUI
    SCLOG(INFO) << "start gui...";
    PSLAM::GraphSLAMGUI gui(&graphSlam, dataset_loader_.get());
    if(params.use_render) gui.SetRender(dataset_loader_->GetCamParamDepth().width,dataset_loader_->GetCamParamDepth().height,path, true);
    if(!params.scene_graph_json_dir.empty()) gui.SetSceneGraphJsonDir(params.scene_graph_json_dir);
#ifdef COMPILE_WITH_ROS_PUBLISH
    if(!params.scene_graph_ros_topic.empty()) gui.SetSceneGraphRosTopic(params.scene_graph_ros_topic);
#endif
    const auto gui_start = Clock::now();
    gui.run();
    if (configPslam.tele_yolo_object_graph) {
        const std::string freq_dir = !params.scene_graph_json_dir.empty()
                                     ? params.scene_graph_json_dir
                                     : params.pth_out;
        if (!freq_dir.empty()) {
            const int    n_frames  = graphSlam.GetProcessedFrameCount();
            const double proc_ms   = graphSlam.GetProcessingTimeMs();
            tools::PathTool::create_folder(freq_dir);
            std::ofstream freq_file(WithTrailingSlash(freq_dir) + "scene_graph_freq.txt");
            freq_file << std::fixed << std::setprecision(3)
                      << "total_frames:  " << n_frames                       << "\n"
                      << "frame_rate_hz: " << (n_frames / (proc_ms * 1e-3))  << "\n";
        }
    }
#else
#ifdef COMPILE_WITH_ROS_PUBLISH
    std::unique_ptr<PSLAM::SceneGraphRosPublisher> ros_publisher;
    if (!params.scene_graph_ros_topic.empty())
        ros_publisher = std::make_unique<PSLAM::SceneGraphRosPublisher>(params.scene_graph_ros_topic);
#endif
    SCLOG(INFO) << "start processing frames...";
    end_to_end_start = Clock::now();
    CTICK("[APP]0.end_to_end_process");
    while (true) {
        CTICK("[APP]2.retrieve");
        if (!dataset_loader_->Retrieve()){CTOCK("[APP]2.retrieve");break;}
        CTOCK("[APP]2.retrieve");
        ++processed_frame_count;
        SCLOG(VERBOSE) << "process frame: " << dataset_loader_->GetFrameIndex();
        const Eigen::Matrix4f pose = dataset_loader_->GetPose().inverse();
        auto rgb = dataset_loader_->GetRGBImage();
        auto d   = dataset_loader_->GetDepthImage();
        const Eigen::Matrix4f *network_pose =
                dataset_loader_->HasNetworkPose() ? &dataset_loader_->GetNetworkPose() : nullptr;

        if(renderer) {
            auto t_p = dataset_loader_->GetPose();
            t_p.topRightCorner<3, 1>() /= 1000.f;
            t_p.transposeInPlace();
            cv::Mat t_rgb;
            renderer->Render(t_rgb,d,t_p,dataset_loader_->GetCamParamDepth());
        }
        CTICK("[ALL]0.all");
        graphSlam.ProcessFrame(dataset_loader_->GetFrameIndex(), rgb, d, &pose, network_pose);
        CTOCK("[ALL]0.all");

        if (!params.scene_graph_json_dir.empty()) {
            WriteSceneGraphJson(dataset_loader_->GetFrameIndex(),
                                graphSlam.GetGraph(),
                                params.scene_graph_json_dir,
                                &pose);
        }
#ifdef COMPILE_WITH_ROS_PUBLISH
        if (ros_publisher) {
            ros_publisher->PublishDelta(dataset_loader_->GetFrameIndex(), graphSlam.GetGraph(), &pose);
        }
#endif

    }
    SCLOG(INFO) << "frame processing finished.";

    // Save scene graph generation frequency stats.
    // Priority: scene_graph_json_dir > pth_out.
    if (configPslam.tele_yolo_object_graph) {
        const std::string freq_dir = !params.scene_graph_json_dir.empty()
                                     ? params.scene_graph_json_dir
                                     : params.pth_out;
        if (!freq_dir.empty()) {
            const int    n_frames = graphSlam.GetProcessedFrameCount();
            tools::PathTool::create_folder(freq_dir);
            std::ofstream freq_file(WithTrailingSlash(freq_dir) + "scene_graph_freq.txt");
            const double proc_s = graphSlam.GetProcessingTimeMs() * 1e-3;
            freq_file << std::fixed << std::setprecision(3)
                      << "total_frames:    " << n_frames                                          << "\n"
                      << "frame_rate_hz:   " << (n_frames / proc_s)                              << "\n"
                      << "yolo_update_hz:  " << (graphSlam.GetSceneGraphUpdateCount() / proc_s)  << "\n"
                      << "yolo_infer_hz:   " << graphSlam.GetYoloHz()                            << "\n";
        }
    }
#endif
    if(params.save || params.save_time) {
        CTICK("[ALL]1.StopGraphPredictor");
        graphSlam.Stop();
        CTOCK("[ALL]1.StopGraphPredictor");
    }

#ifndef COMPILE_WITH_PSLAM_GUI
    CTOCK("[APP]0.end_to_end_process");
    end_to_end_end = Clock::now();
    end_to_end_wall_ms = std::chrono::duration<double, std::milli>(end_to_end_end - end_to_end_start).count();
#endif

    if(params.save) {
        SCLOG(INFO) << "saving results.";
        if(params.save_graph_ply) {
            auto colorMode = params.use_predict? PSLAM::GraphSLAM::SAVECOLORMODE_INSTANCE : PSLAM::GraphSLAM::SAVECOLORMODE_RGB;
            if(params.use_predict)
                graphSlam.SaveNodesToPLY(params.segment_filter, params.pth_out, colorMode,
                                     params.binary);
        }

        if(params.save_surfel_ply)
            graphSlam.SaveSurfelsToPLY(params.pth_out,params.save_name,graphSlam.GetInSeg()->map().surfels,params.binary);

        if(params.save_graph) {
            std::string scan_id;
            if (isTeleDataset)
                scan_id = tools::PathTool::getFileName(path);
            else
                scan_id = tools::PathTool::getFileName(tools::PathTool::find_parent_folder(path, 1));
            auto predictions = graphSlam.GetSceneGraph(params.full_prop);
            json11::Json::object json;
            json[scan_id] = predictions;
            ORUtils::JsonUtils::Dump(json, params.pth_out + "/predictions.json");
        }
    }

    if(params.save_time) {
        const std::string stats_dir = ".";
        const std::string stats_prefix = WithTrailingSlash(stats_dir);
        SCSLAM::EVALUATION::Logging::printResultToFile(stats_dir, "times.txt");
#ifndef COMPILE_WITH_PSLAM_GUI
        if (processed_frame_count > 0) {
            std::fstream file(stats_prefix + "times.txt", std::ios::app);
            const double avg_frame_ms = end_to_end_wall_ms / static_cast<double>(processed_frame_count);
            file << "[APP]1.avg_per_image_end_to_end_async;" << std::to_string(avg_frame_ms)
                 << ";0.000000;0.000000;" << std::to_string(processed_frame_count)
                 << ";" << std::to_string(end_to_end_wall_ms) << "\n";
        }
#endif
#ifdef COMPILE_WITH_GRAPHPRED
        if(graphSlam.GetGraphPred()) {
            const auto &times = graphSlam.GetGraphPred()->GetTimes();
            for (const auto &pair: times) {
                std::fstream file(stats_prefix + "times_graph_" + pair.first + ".txt", std::ios::out);
                file << pair.first << "\n";
                for (const auto &pp : pair.second)
                    file << pp.first << "," << pp.second << "\n";
                file << "\n";
                file.close();
            }
        }
#endif
    }

#ifdef COMPILE_WITH_ROS_PUBLISH
    if (rclcpp::ok()) rclcpp::shutdown();
#endif

    return 0;
}
