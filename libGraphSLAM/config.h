//
// Created by sc on 10/9/20.
//

#ifndef GRAPHSLAM_CONFIG_H
#define GRAPHSLAM_CONFIG_H
#include <string>
#include <fstream>
#include <inseg_lib/inseg_config.h>
#include <inseg_lib/main_config.h>
#include <inseg_lib/map_config.h>
#include <inseg_lib/segmentation_config.h>
//#include <inseg_lib/segmentation.h>
namespace PSLAM {
    struct ConfigPSLAM {
        ConfigPSLAM();
        explicit ConfigPSLAM(const std::string &path);
        ~ConfigPSLAM();
        bool use_thread;

        bool use_fusion;

        /// Use graph predict or not
        bool graph_predict;

        /// model path
        std::string pth_model;

        /// dataset root, used by model adapters to find class/relationship files
        std::string pth_dataset;

        int filter_num_node;

        /// Keep the previous node label when the fused top-1 class probability is below this value.
        /// A value <= 0 disables the filter.
        float label_confidence_threshold;

        /// Legacy Tele ONNX compatibility for models trained before the quaternion/camera-basis
        /// fix. New fixed-quaternion models should use the reconstruction geometry directly.
        bool tele_network_geometry_compatibility;

        /// Optional folder for Tele ONNX debug dumps (visual bboxes, crops, feature norms).
        std::string tele_debug_dir;

        /// Tele node label post-processing: skip node prediction below this top-1 probability.
        float tele_no_prediction_threshold;

        /// Tele node label post-processing: use location/category priors below this top-1 probability.
        float tele_prior_confidence_threshold;

        /// Tele node label post-processing: number of class candidates checked by priors.
        int tele_prior_top_k;

        /// Optional YOLO segmentation ONNX used by Tele relation-only models.
        std::string tele_yolo_model_path;
        float tele_yolo_conf_threshold;
        float tele_yolo_iou_threshold;
        int tele_yolo_min_mask_area;
        int tele_yolo_min_depth_points;
        float tele_yolo_node_overlap_threshold;
        bool tele_yolo_object_graph;
        float tele_yolo_track_assoc_distance;
        int tele_yolo_track_max_age;
        int tele_yolo_object_min_points;
        int tele_yolo_object_sample_stride;
        int tele_yolo_object_max_points_per_mask;
        int tele_yolo_track_max_points;
        float tele_yolo_object_voxel_size;
        float tele_yolo_vis_voxel_size;    // unused (kept for ABI compat)
        int   tele_yolo_vis_max_points;   // max VisPoints per track
        int   tele_yolo_vis_sample_stride;// depth-image stride for vis scan (1 = every pixel)

//        int predict_when_have_at_least_n_node;
        size_t n_pts;

        float neighbor_margin; // mm

        /// Only update the information of a node if the node has size change to N percent
        float update_thres_node_size;
        /// Update a node if its time stamp is older than n
        int update_thres_time_stamp;

        std::string pth;


        inseg_lib::InSegConfig inseg_config;
        inseg_lib::MainConfig main_config;
        inseg_lib::MapConfig map_config;
        inseg_lib::SegmentationConfig segmentation_config;
    };


}

#endif //GRAPHSLAM_CONFIG_H
