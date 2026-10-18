#include "node.h"

// #include "inseg_lib/surfel.h"
#include <ORUtils/Logging.h>
#include <random>
#include <algorithm>
using namespace PSLAM;

Node::Node(int label):
idx(label), instance_idx(0), mDebug(false), time_stamp(0),lastUpdatePropertySize(0),
mbNeedUpdateNodeFeature(false),mIdxCounter(0) {
    last_class_predicted = Unknown();
    instance_idx=idx;
    centroid.setZero();
    pos_sum.setZero();
    bbox_max.setZero();
    bbox_min.setZero();
}

int Node::Add(const SurfelPtr &surfel) {
    assert(surfel->GetLabel() == this->idx);
    /*Find Idx*/
    int idx_;
    {
        Lock lock(mMutNode);
        if(!mqFreeIndices.empty()) {
            idx_ = mqFreeIndices.front();
            mqFreeIndices.pop();
        } else {
            idx_ = mIdxCounter++;
        }
    }

    size_t size;
    {
        Lock lock(mMutSurfel);
        surfels.insert( {idx_, surfel} );
        size = surfels.size();
    }

    /*Update Properties*/
    {
        Lock lock(mMutNode);
        pos_sum += surfel->pos;
        centroid = pos_sum / size;
        if(size == 1){
            bbox_min=bbox_max=surfel->pos;
        } else
            for(size_t j=0;j<3;++j) {
                if (surfel->pos[j] > bbox_max[j]) {
                    bNeedCheckNeighbor=true;
                    bbox_max[j] = surfel->pos[j];
                }
                if (surfel->pos[j] < bbox_min[j]) {
                    bNeedCheckNeighbor=true;
                    bbox_min[j] = surfel->pos[j];
                }
            }
    }
    return idx_;
}

void Node::Remove(const int index_in_graph_old) {
    size_t size;
    SurfelPtr surfel;
    {
        Lock lock_surfel(mMutSurfel);
        size = surfels.size();
        if(surfels.find(index_in_graph_old) == surfels.end())
            throw std::runtime_error("Trying to remove surfel but it doesn't exist.");
        assert(surfels.at(index_in_graph_old)->label == this->idx);
        surfel = surfels.at(index_in_graph_old);
        surfels.erase(index_in_graph_old);
    }

    Lock lock_node(mMutNode);
    pos_sum -= surfel->pos;
    centroid = pos_sum / size;
    bool need_recal_bbox = surfel->pos==bbox_min || surfel->pos ==bbox_max;

    mqFreeIndices.emplace(index_in_graph_old);
    if(need_recal_bbox) {
        Lock lock_surfel(mMutSurfel);
        for (size_t i=0;i<size;++i){
            auto surfel_ = surfels.at(i);
            if(i==0){
                bbox_max = bbox_min = surfel_->pos;
                continue;
            }
            for (size_t j = 0; j < 3; ++j) {
                if (surfel_->pos[j] > bbox_max[j]) bbox_max[j] = surfel_->pos[j];
                if (surfel_->pos[j] < bbox_min[j]) bbox_min[j] = surfel_->pos[j];
            }
        }
        bNeedCheckNeighbor = true;
    }
}

void Node::Update(const int label,
                   const int index,
                   const Eigen::Vector3f& pos,
                   const Eigen::Vector3f& normal,
                   const std::array<unsigned char, 3>& color) {
    throw std::runtime_error("not implemented.");
}

void Node::RemoveEdge(const EdgePtr &edge){
    Lock lock(mMutEdge);
    if (edges.find(edge) == edges.end()) return;
    edges.erase(edge);
}

void Node::UpdatePrediction(const std::map<std::string, float> &pd,
                            const std::map<std::string,std::pair<size_t, size_t>> &sizeAndEdge,
                            bool fusion,
                            float label_confidence_threshold) {
    Lock lock(mMutPred);
    for(const auto &pair:pd) {
        auto name = pair.first;
        auto value = pair.second;

        if(mClsProb.find(name) == mClsProb.end()) {
            mClsProb[name] = value;
            if (sizeAndEdge.find(name) == sizeAndEdge.end())
                SCLOG(ERROR) << "cannot find " << name << " in sizeAndEdge";
            mSizeAndEdge[name] = sizeAndEdge.at(name);
            mClsWeight[name] = 1;
        }
        else {
            if(fusion) {
                static const float max_w = 100;
                const static float new_w = 1;
                const auto &old_value = mClsProb.at(name);
                const auto &old_w = mClsWeight.at(name);
                mClsProb.at(name) = (old_value * old_w + value * new_w) / (old_w + new_w);
                mClsWeight.at(name) = std::min(max_w, mClsWeight.at(name) + new_w);
            } else {
                // simply over-write
                mClsProb[name] = value;
            }
        }
    }
    // find the newest with maximum prob
    if(mClsProb.empty()) return;
    {
        float max_v=mClsProb.begin()->second;
        std::string max_label = GetLabel();
        for(const auto& pair:mClsProb){
            if(pair.second>=max_v){
                max_v = pair.second;
                max_label = pair.first;
            }
        }
        Lock lock_pd (mMutPDLabel);
        if (label_confidence_threshold <= 0.0f || max_v >= label_confidence_threshold)
            last_class_predicted = max_label;
    }
}

void Node::ClearPrediction() {
    {
        Lock lock(mMutPred);
        mClsProb.clear();
        mClsWeight.clear();
        mSizeAndEdge.clear();
    }
    {
        Lock lock_pd(mMutPDLabel);
        last_class_predicted = Unknown();
    }
    instance_idx = idx;
}

bool Node::CheckConnectivity(Node *nodeP, float margin, bool modify){
    int idx_s = this->idx;
    int idx_t = nodeP->idx;
    if(idx_s == idx_t) return true;
    bool has_out = false;
    {
        Lock lock_this(this->mMutNode);
        for(size_t m=0;m<3;++m){
            if (bbox_min[m]-margin > nodeP->bbox_max[m]+margin) has_out = true;
            if (nodeP->bbox_min[m]-margin > bbox_max[m]+margin) has_out = true;
        }
        if(modify) this->bNeedCheckNeighbor = nodeP->bNeedCheckNeighbor = false;
    }
    if(!modify) return !has_out;

    Lock lock(mMutNN);
    if (has_out) {
        if(neighbors.find(idx_t) != neighbors.end()) neighbors.erase(idx_t);
        if(nodeP->neighbors.find(idx_s) != nodeP->neighbors.end())
            nodeP->neighbors.erase(idx_s);
    } else {
        neighbors.insert(idx_t);
        nodeP->neighbors.insert(idx_s);
    }
    return !has_out;
}

int Node::GetPointSize() {
    Lock lock (mMutSurfel);
    return surfels.size();
}

void Node::UpdateSelectedNode(const size_t time, const size_t filter_size, const size_t num_pts, bool force) {
    if(mbNeedUpdateNodeFeature) return;
    const auto& last_size = lastUpdatePropertySize; // this will be updated in UpdatePrediction
    size_t new_size;
    {
        Lock lock (mMutSurfel);
        new_size = surfels.size();
    }
    static float threshold_size = 0.1;
    static float threshold_time = 60;
    const float changes = abs(float(new_size)-float(last_size))/ float(new_size);
    bool should_update = false;
    should_update |= changes >= threshold_size;
    should_update |= (time - time_stamp) > threshold_time;
    if(force) should_update = true;
    if( !should_update) return;

    Lock lock(mMutSelected);
    std::random_device rd; // obtain a random number from hardware
    std::mt19937 gen(rd()); // seed the generator

    selected_surfels.clear();
    selected_surfels.reserve(num_pts);

    std::vector<SurfelPtr> valid_surfels;
    if (this->surfels.size() < filter_size) {
        return;
    } else {
        valid_surfels.reserve(this->surfels.size());
        for (const auto &surfel:surfels) {
            if (!surfel.second->is_stable || !surfel.second->is_valid) continue;
            valid_surfels.emplace_back(surfel.second);
        }

        if (valid_surfels.size() < filter_size) {
            selected_surfels.clear();
            return;
        }

        if (!mDebug) std::shuffle(valid_surfels.begin(), valid_surfels.end(), gen);
        for (const auto &surfel : valid_surfels) {
            if (selected_surfels.size() == num_pts) break;
            selected_surfels.emplace_back(surfel);
        }

        if (selected_surfels.size() < num_pts) {
            std::uniform_int_distribution<size_t> dist(0, valid_surfels.size() - 1);
            while (selected_surfels.size() < num_pts)
                selected_surfels.emplace_back(valid_surfels.at(dist(gen)));
        }
    }
    lock.unlock();

    if (mDebug)
        for(size_t i=0;i<1;++i) {//TODO: remove me
            Eigen::Vector3f color;
            color.x() = float(selected_surfels.at(i)->color[2])/255*2.f-1;
            color.y() = float(selected_surfels.at(i)->color[1])/255*2.f-1;
            color.z() = float(selected_surfels.at(i)->color[0])/255*2.f-1;
            SCLOG(DEBUG) << selected_surfels.at(i)->pos.transpose() << " " << color.transpose() << " " << selected_surfels.at(i)->normal.transpose();
        }

    // Bbox/centroid/std/orientation are computed from the full valid+stable
    // surfel set (not `selected_surfels`, which is shuffled and truncated/
    // padded to a fixed `num_pts` for the GNN's point-cloud input only).
    // Using the truncated subset here made the box shrink inward from the
    // true extent for any object with many more than num_pts surfels, since
    // uniform random sampling systematically misses the extreme/boundary
    // points (worse the larger/more elongated the object).
    mCentroid.setZero();
    mStd.setZero();
    for(size_t i=0;i<valid_surfels.size();++i){
        const auto &surfel = valid_surfels.at(i);
        mCentroid+=surfel->pos;

        if(i==0){
            mBBox_max=mBBox_min=surfel->pos;
        } else {
            for(size_t j=0;j<3;++j) {
                if (surfel->pos[j] > mBBox_max[j]) {
                    mBBox_max[j] = surfel->pos[j];
                }
                if (surfel->pos[j] < mBBox_min[j]) {
                    mBBox_min[j] = surfel->pos[j];
                }
            }
        }
    }
    mCentroid/=valid_surfels.size();

    for(auto & valid_surfel : valid_surfels) {
        for (size_t d = 0; d < 3; ++d)
            mStd[d] += pow(valid_surfel->pos[d] - mCentroid[d], 2);
    }
    for (size_t d = 0; d < 3; ++d)
        mStd[d] = std::sqrt(mStd[d] / (valid_surfels.size() - 1.f) );

    // 2D PCA on XZ plane (Y is vertical) to estimate object orientation.
    if (valid_surfels.size() >= 5) {
        Eigen::Matrix2f cov = Eigen::Matrix2f::Zero();
        for (const auto &s : valid_surfels) {
            const float dx = s->pos[0] - mCentroid[0];
            const float dz = s->pos[2] - mCentroid[2];
            cov(0,0) += dx*dx; cov(0,1) += dx*dz;
            cov(1,0) += dz*dx; cov(1,1) += dz*dz;
        }
        cov /= static_cast<float>(valid_surfels.size());
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix2f> solver(cov);
        // col(1) = eigenvector of largest eigenvalue = primary axis
        const Eigen::Vector2f e1 = solver.eigenvectors().col(1);
        mOrientAxis1 = Eigen::Vector3f(e1[0], 0.f, e1[1]).normalized();
        mHasOrientation = true;
    } else {
        mHasOrientation = false;
    }
    mHasBBox = true;

    //TODO: remove me
    if (mDebug)
    {
        float volume = 1;
        auto dim = (mBBox_max-mBBox_min)*1e-3;
        for (size_t d = 0; d < 3; ++d) volume *= dim[d];
        float length = *std::max_element(dim.begin(), dim.end());
        size_t valid_surfel_count = 0;
        for(const auto& surfel:surfels)
            if(surfel.second->is_valid && surfel.second->is_stable) valid_surfel_count++;
    }

    mbNeedUpdateNodeFeature=true;
    lastUpdatePropertySize = new_size;
    time_stamp = time;
}

const std::string Node::GetLabel() const {
    Lock lock(mMutPDLabel);
    return last_class_predicted;
}
