//
// Tele dataset loader.
//

#ifndef LIBSURFELRECONSTRUCTION_DATALOADER_TELE_H
#define LIBSURFELRECONSTRUCTION_DATALOADER_TELE_H

#include "dataset_loader.h"
#include <string>
#include <vector>

namespace PSLAM {
    class DatasetLoader_Tele : public DatasetLoader_base {
    public:
        explicit DatasetLoader_Tele(std::shared_ptr<DatasetDefinitionBase> dataset);
        void Reset() override;
        bool Retrieve() override;
        [[nodiscard]] const Eigen::Matrix4f& GetNetworkPose() const override { return m_network_pose; }
        [[nodiscard]] bool HasNetworkPose() const override { return true; }

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    private:
        struct FrameRecord {
            int frame_number = 0;
            std::string rgb;
            std::string depth;
            std::string pose;
        };

        std::vector<FrameRecord> m_frames;
        size_t m_cursor = 0;
        Eigen::Matrix4f m_network_pose = Eigen::Matrix4f::Identity();

        void ScanFrames();
        void InitializeCameraFromFirstFrame();
        bool TryLoadIntrinsicsFile(int image_width, int image_height);
        void LoadPoseJson(Eigen::Matrix4f &pose, const std::string &path) const;
        void LoadNetworkPoseJson(Eigen::Matrix4f &pose, const std::string &path) const;
    };
}

#endif //LIBSURFELRECONSTRUCTION_DATALOADER_TELE_H
