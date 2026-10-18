//
// Created by sc on 11/4/20.
//
#include "config.h"
using namespace PSLAM;
ConfigPSLAM::ConfigPSLAM(const std::string &path){
    if(!path.empty()) pth=path;
    std::fstream f(pth,std::ios::in);
    if(!f.is_open())return;
}

ConfigPSLAM::~ConfigPSLAM(){}

ConfigPSLAM::ConfigPSLAM() {
    use_thread = true;

    use_fusion = true;

    /// Use graph predict or not
    graph_predict = true;

    /// model path
    pth_model = "/home/sc/research/PersistentSLAM/python/3DSSG/experiments/exp10_2_1/traced/";
    pth_model = "/home/sc/research/PersistentSLAM/python/3DSSG/experiments/exp10_4/traced/";
    pth_model = "/home/sc/research/PersistentSLAM/python/3DSSG/experiments/exp10_5/traced/";
    pth_dataset = "";

    filter_num_node = 512;
    label_confidence_threshold = 0.0f;
    tele_network_geometry_compatibility = false;
    tele_debug_dir = "";
    tele_no_prediction_threshold = 0.10f;
    tele_prior_confidence_threshold = 0.30f;
    tele_prior_top_k = 3;
    tele_yolo_model_path = "";
    tele_yolo_conf_threshold = 0.35f;
    tele_yolo_iou_threshold = 0.70f;
    tele_yolo_min_mask_area = 64;
    tele_yolo_min_depth_points = 50;
    tele_yolo_node_overlap_threshold = 0.05f;
    tele_yolo_object_graph = false;
    tele_yolo_track_assoc_distance = 0.60f;
    tele_yolo_track_max_age = -1;
    tele_yolo_object_min_points = 50;
    tele_yolo_object_sample_stride = 2;
    tele_yolo_object_max_points_per_mask = 2000;
    tele_yolo_track_max_points = 200000;
    tele_yolo_object_voxel_size = 0.030f;
    tele_yolo_vis_voxel_size = 0.005f;  // unused
    tele_yolo_vis_max_points = 200000;
    tele_yolo_vis_sample_stride = 1;

    n_pts = 512;

    neighbor_margin=500; // mm

    /// Only update the information of a node if the node has size change to N percent
    update_thres_node_size = 0.2;
    /// Update a node if its time stamp is older than n
    update_thres_time_stamp = 50;


    pth = "./config_graph.txt";
}
