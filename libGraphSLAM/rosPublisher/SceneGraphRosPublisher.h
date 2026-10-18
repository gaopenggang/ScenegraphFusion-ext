#pragma once
#ifdef COMPILE_WITH_ROS_PUBLISH
#include <Eigen/Core>
#include <memory>
#include <string>

#include "../graph.h"

namespace PSLAM {

// Publishes the scene graph as a scene_msgs/TeleScene ROS2 topic. Each call
// to PublishDelta() diffs the graph against the state seen on the previous
// call and publishes only the nodes/edges that were added, changed, or
// removed (ope_id / old_*+new_* fields), so message size tracks how much the
// scene changed rather than the total node/edge count. Nothing is published
// when there is no change.
class SceneGraphRosPublisher {
public:
    explicit SceneGraphRosPublisher(const std::string &topic_name);
    ~SceneGraphRosPublisher();

    SceneGraphRosPublisher(const SceneGraphRosPublisher &) = delete;
    SceneGraphRosPublisher &operator=(const SceneGraphRosPublisher &) = delete;

    void PublishDelta(int frame_idx, Graph *g, const Eigen::Matrix4f *pose = nullptr);

private:
    struct Impl;
    std::unique_ptr<Impl> mImpl;
};

}  // namespace PSLAM
#endif  // COMPILE_WITH_ROS_PUBLISH
