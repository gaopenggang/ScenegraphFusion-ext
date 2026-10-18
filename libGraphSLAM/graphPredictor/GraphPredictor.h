//
// Created by sc on 8/24/20.
//

#pragma once
//#include "PointNetGCN.h"
#include <memory>
#include <inseg_lib/lib.h>
#include <random>
#include "../node.h"
#include "../edge.h"
//#include "../nodeProperties.h"
#include "../graph.h"
#include "../config.h"
#include "../../Objects/Camera/CameraParameters.h"
#include <json11.hpp>
#include "EatGCN.h"
#include <future>
#include <opencv2/core.hpp>
#include <unordered_map>
#include "../VisPoint.h"

namespace PSLAM {
#ifdef COMPILE_WITH_ONNX
    struct TeleYoloDetection {
        cv::Rect box;
        cv::Mat  mask;
        int      class_idx = -1;
        float    score     = 0.f;
    };

    typedef std::tuple<MemoryBlock3D,MemoryBlock2D,MemoryBlock2D, std::map<size_t,size_t>,std::map<int, std::pair<size_t,size_t>> > GCNINPUTDATA;
    class GraphPredictor {
    public:
        typedef std::tuple< std::map<size_t,std::map<std::string,float>>, std::vector<EdgePtr>,std::map<int, std::pair<size_t,size_t>> > Prediction;

        explicit GraphPredictor( const ConfigPSLAM *configPslam);
        ~GraphPredictor();
//        std::tuple<std::map<size_t,size_t>,std::vector<Edge>> Predict(std::vector<std::shared_ptr<inseg_lib::Surfel>> &surfels);
        void RUN(Graph *graph);

        void SetVisualFrame(const cv::Mat &rgb,
                            const cv::Mat &depth,
                            const Eigen::Matrix4f &pose,
                            const CameraParameters &cam,
                            const Eigen::Matrix4f *tele_network_pose = nullptr,
                            int frame_idx = -1);

        void Stop();

        bool UpdateTeleYoloObjectGraph(Graph *object_graph, int frame_idx);

        const cv::Mat& GetYoloVisualization() const { return mYoloVisualization; }

        void SetThread(bool option);

        /// Check if thread is dead
        bool Pin();

        bool Updated();

        void SetUpdate(bool option);

        std::shared_ptr<Prediction> Predict(
                const std::shared_ptr<GCNINPUTDATA>& data);

        const std::map<std::string, json11::Json>& GetParams();

        std::map<std::string,std::vector<std::pair<int,float>>> GetTimes();

        bool mbVerbose=false;
        std::map<size_t,std::string> mLabels, mRelationships;
        std::map<std::string,size_t> mLabelsName2Idx,mRelationshipsName2Idx;
    private:
        const ConfigPSLAM *mConfigPslam;
        std::unique_ptr<Ort::MemoryInfo> mMemoryInfo;
        std::unique_ptr<EatGCN> mEatGCN;
        bool mUseTeleModules = false;
        bool mUseTeleVisualEncoder = false;
        bool mTeleHasForegroundFilter = false;
        bool mTeleHasObjPredictor = false;
        bool mTeleRelationOnly = false;
        bool mUseTeleYolo = false;
        std::string mTeleModulePath;
        std::unique_ptr<Ort::Env> mTeleOrtEnv;
        std::map<std::string, std::unique_ptr<Ort::Session>> mTeleSessions;
        std::map<std::string, json11::Json> mTeleParams;
        cv::Mat mVisualRGB;
        cv::Mat mVisualDepth;
        Eigen::Matrix4f mVisualPose = Eigen::Matrix4f::Identity();
        Eigen::Matrix4f mTeleNetworkPose = Eigen::Matrix4f::Identity();
        CameraParameters mVisualCam{};
        bool mHasVisualFrame = false;
        bool mHasTeleNetworkPose = false;
        int mVisualFrameIndex = -1;
        std::mutex mMutVisualFrame;
        Graph *mpGraph = nullptr;
        std::future<void> mThread;
        bool mbShouldStop = false;
        bool mbThread = true;
        bool source_to_target;
        bool mDebug = false;
        std::atomic_bool mUpdated;
        void Process_IMPL();
        void InitTeleModules(const std::string &path_onnx);
        void LoadTeleLabels(const std::string &path_onnx);
        std::vector<Ort::Value> RunTeleModule(const std::string &name,
                                              std::vector<Ort::Value> &inputs) const;
        MemoryBlock2D BuildTeleVisualFeature(const std::vector<NodePtr> &nodes,
                                             size_t visual_dim);
        bool GetTeleNetworkGeometryTransform(Eigen::Matrix3f &rotation,
                                             Eigen::Vector3f &translation);
        bool ComputeTeleFrameSceneBounds(Eigen::Vector3f &scene_min,
                                         Eigen::Vector3f &scene_max,
                                         const Eigen::Matrix3f &geometry_rotation,
                                         const Eigen::Vector3f &geometry_translation,
                                         bool use_geometry_transform);
        void ProcessTele_IMPL(const std::map<int,NodePtr> &selected_nodes,
                              const std::map<int,std::pair<size_t,size_t>> &sizeAndEdge);
        void UpdateNodeFeature(const std::map<int,NodePtr> &selected_nodes);
        void UpdateEdgeFeature(const std::map<int,NodePtr> &selected_nodes);
        void UpdateGCNFeature(const std::map<int,NodePtr> &selected_nodes, size_t level);
        void UpdatePrediction(const std::map<int,NodePtr> &selected_nodes,
                const std::map<int,std::pair<size_t,size_t>> &sizeAndEdge);
        void UpdateShapeFeature(const std::vector<NodePtr> &selected_nodes);

        std::map<int, NodePtr> getNeighbor(const std::map<int,NodePtr> &vNodes, size_t min_pts);

        std::map<std::string,std::vector<std::pair<int,float>>> mTimes;

        // YOLO async thread: submit current frame, get back paired (detections + depth).
        struct YoloAsyncInput {
            cv::Mat rgb, depth;
            Eigen::Matrix4f world_to_camera = Eigen::Matrix4f::Identity();
            CameraParameters cam{};
            int frame_idx = -1;
        };
        struct YoloAsyncResult {
            std::vector<TeleYoloDetection> detections;
            cv::Mat rgb, depth, visualization;
            Eigen::Matrix4f world_to_camera = Eigen::Matrix4f::Identity();
            CameraParameters cam{};
            int frame_idx = -1;
        };
        YoloAsyncInput          mYoloInput;
        std::mutex              mMutYoloInput;
        std::condition_variable mCvYoloInput;
        bool                    mYoloInputReady = false;
        YoloAsyncResult         mYoloResult;
        std::mutex              mMutYoloResult;
        std::condition_variable mCvYoloResult;
        bool                    mYoloResultReady    = false;
        bool                    mFirstYoloReceived  = false;
        double                  mYoloInferTotalMs   = 0.0;
        int                     mYoloInferCount     = 0;
        std::future<void>       mYoloThread;
        std::atomic<bool>       mYoloThreadStop{false};
        void                    YoloThread_IMPL();

        struct TeleYoloObjectTrack {
            int graph_id = -1;
            int class_idx = -1;
            float score = 0.f;
            int last_seen_frame = -1;
            Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
            Eigen::Vector3f bbox_min = Eigen::Vector3f::Zero();
            Eigen::Vector3f bbox_max = Eigen::Vector3f::Zero();
            std::unordered_map<uint64_t, SurfelPtr> surfels; // GNN/graph (coarse voxels, deduped)
            VisPointVec vis_surfels;                          // display only (flat, lightweight)
        };
        std::map<int, TeleYoloObjectTrack> mTeleYoloTracks;
        int mTeleYoloNextTrackId  = 1;
        int mLastYoloFrameIdx     = -1;
    public:
        int    GetLastYoloFrameIdx() const { return mLastYoloFrameIdx; }
        double GetYoloInferTotalMs() const { return mYoloInferTotalMs; }
        int    GetYoloInferCount()   const { return mYoloInferCount; }
    private:
        cv::Mat    mYoloVisualization;
        VisPointVec mTeleYoloVisSurfels;
        std::mutex  mMutVisSurfels;
        size_t      mTeleYoloTotalVisCount = 0; // last snapshot size; skip rebuild when unchanged
    public:
        // Returns a copy only when the snapshot has grown/shrunk since last call.
        // Caller passes its own cached count; returns empty vector if unchanged.
        VisPointVec GetTeleYoloVisSurfels(size_t &caller_count) {
            std::unique_lock<std::mutex> lock(mMutVisSurfels);
            if (mTeleYoloVisSurfels.size() == caller_count) return {};
            caller_count = mTeleYoloVisSurfels.size();
            return mTeleYoloVisSurfels;
        }
    };
#else
    class GraphPredictor {
    public:
    public:
        explicit GraphPredictor(const std::string &path_onnx){};

        std::tuple<std::map<size_t,size_t>,std::vector<Edge>> Predict(std::vector<std::shared_ptr<inseg_lib::Surfel>> &surfels){};

        bool mbVerbose=true;
        std::map<size_t,std::string> mLabels, mRelationships;
    private:
    };
#endif
    typedef std::shared_ptr<GraphPredictor> GraphPredictorPtr;
    static inline GraphPredictorPtr MakeGraphPredictor(const ConfigPSLAM *configPslam) {return std::make_shared<GraphPredictor>(configPslam);}
}
