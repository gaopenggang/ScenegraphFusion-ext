//
// Tele dataset support.
//

#ifndef LIBSURFELRECONSTRUCTION_DATASETTELE_H
#define LIBSURFELRECONSTRUCTION_DATASETTELE_H

#include "dataset_base.h"
#include <utility>

namespace PSLAM {
    class TeleDataset : public DatasetDefinitionBase {
    public:
        TeleDataset(INPUTE_TYPE type, std::string path) {
            datasetType = type;
            if (!path.empty() && path.back() == '/') path.pop_back();
            folder = std::move(path);

            rotate_pose_img = false;
            frame_index_counter = 1;
            number_length = 5;
            number_pose = 5;
            max_depth = 10000;

            folder_rgb = "/rgb";
            folder_depth = "/depth";
            folder_pose = "/pose";
            prefix_rgb = "/head_rgb_";
            prefix_depth = "/head_depth_";
            prefix_pose = "/pose_";
            suffix_rgb = ".png";
            suffix_depth = ".png";
            suffix_pose = ".json";

            min_pyr_level = 2;
        }
    };
}

#endif //LIBSURFELRECONSTRUCTION_DATASETTELE_H
