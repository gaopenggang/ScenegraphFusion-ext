//
// Tele dataset loader.
//

#include "../include/dataLoader/dataloader_tele.h"

#include <ORUtils/JsonUtil.h>
#include <algorithm>
#include <cmath>
#include <dirent.h>
#include <fstream>
#include <regex>
#include <stdexcept>
#include <sys/stat.h>

using namespace PSLAM;

namespace {
    bool FileExists(const std::string &path) {
        std::ifstream file(path);
        return file.good();
    }

    bool DirectoryExists(const std::string &path) {
        struct stat sb{};
        return stat(path.c_str(), &sb) == 0 && S_ISDIR(sb.st_mode);
    }

    int ParseLastNumber(const std::string &name) {
        std::regex number_regex("(\\d+)");
        int value = -1;
        for (auto it = std::sregex_iterator(name.begin(), name.end(), number_regex);
             it != std::sregex_iterator(); ++it) {
            value = std::stoi((*it)[1].str());
        }
        return value;
    }

    std::vector<std::string> ListPngFiles(const std::string &folder) {
        std::vector<std::string> files;
        DIR *dir = opendir(folder.c_str());
        if (!dir) return files;
        while (auto *entry = readdir(dir)) {
            std::string name = entry->d_name;
            if (name == "." || name == "..") continue;
            if (name.size() < 4 || name.substr(name.size() - 4) != ".png") continue;
            files.emplace_back(name);
        }
        closedir(dir);
        std::sort(files.begin(), files.end(), [](const std::string &a, const std::string &b) {
            const int ia = ParseLastNumber(a);
            const int ib = ParseLastNumber(b);
            if (ia >= 0 && ib >= 0 && ia != ib) return ia < ib;
            return a < b;
        });
        return files;
    }

    std::string ParentFolder(std::string path) {
        if (!path.empty() && path.back() == '/') path.pop_back();
        const auto pos = path.find_last_of('/');
        if (pos == std::string::npos) return "";
        if (pos == 0) return "/";
        return path.substr(0, pos);
    }
}

DatasetLoader_Tele::DatasetLoader_Tele(std::shared_ptr<DatasetDefinitionBase> dataset):
        DatasetLoader_base(std::move(dataset)) {
    ScanFrames();
    InitializeCameraFromFirstFrame();
    Reset();
}

void DatasetLoader_Tele::ScanFrames() {
    const std::string rgb_dir = m_dataset->folder + "/rgb";
    const std::string depth_dir = m_dataset->folder + "/depth";
    const std::string pose_dir = m_dataset->folder + "/pose";
    if (!DirectoryExists(rgb_dir) || !DirectoryExists(depth_dir) || !DirectoryExists(pose_dir)) {
        throw std::runtime_error("Tele dataset expects rgb/depth/pose folders under: " + m_dataset->folder);
    }

    const auto rgb_files = ListPngFiles(rgb_dir);
    for (const auto &rgb_file: rgb_files) {
        const int frame_number = ParseLastNumber(rgb_file);
        if (frame_number < 0) continue;

        const auto number_begin = rgb_file.find_last_of('_') + 1;
        const auto number_end = rgb_file.find_last_of('.');
        const std::string number = rgb_file.substr(number_begin, number_end - number_begin);
        const std::string rgb_path = rgb_dir + "/" + rgb_file;
        const std::string depth_path = depth_dir + "/head_depth_" + number + ".png";
        const std::string pose_path = pose_dir + "/pose_" + number + ".json";
        if (!FileExists(depth_path) || !FileExists(pose_path)) continue;
        m_frames.push_back({frame_number, rgb_path, depth_path, pose_path});
    }

    if (m_frames.empty()) {
        throw std::runtime_error("No Tele frames found under: " + m_dataset->folder);
    }
    frame_index_max = m_frames.back().frame_number;
}

void DatasetLoader_Tele::InitializeCameraFromFirstFrame() {
    const auto rgb = cv::imread(m_frames.front().rgb, cv::IMREAD_COLOR);
    if (rgb.empty()) {
        throw std::runtime_error("Unable to read Tele RGB frame: " + m_frames.front().rgb);
    }
    const int width = rgb.cols;
    const int height = rgb.rows;
    if (TryLoadIntrinsicsFile(width, height)) return;

    const float hfov = static_cast<float>(M_PI) * 0.5f;
    const float aspect = static_cast<float>(height) / static_cast<float>(width);
    const float vfov = 2.0f * std::atan(std::tan(hfov * 0.5f) * aspect);
    const float fx = (static_cast<float>(width) * 0.5f) / std::tan(hfov * 0.5f);
    const float fy = (static_cast<float>(height) * 0.5f) / std::tan(vfov * 0.5f);
    const float cx = static_cast<float>(width) * 0.5f;
    const float cy = static_cast<float>(height) * 0.5f;
    m_cam_param_rgb.Set(width, height, fx, fy, cx, cy, 1.f);
    m_cam_param_d = m_cam_param_rgb;
}

bool DatasetLoader_Tele::TryLoadIntrinsicsFile(int image_width, int image_height) {
    const std::vector<std::string> candidates = {
            m_dataset->folder + "/intrinsics.txt",
            ParentFolder(m_dataset->folder) + "/intrinsics.txt"
    };
    for (const auto &path: candidates) {
        if (!FileExists(path)) continue;
        std::ifstream file(path);
        float fx, fy, cx, cy;
        int width = image_width;
        int height = image_height;
        if (!(file >> fx >> fy >> cx >> cy)) continue;
        if (file >> width >> height) {
            if (width <= 0 || height <= 0) {
                width = image_width;
                height = image_height;
            }
        }
        m_cam_param_rgb.Set(width, height, fx, fy, cx, cy, 1.f);
        m_cam_param_d = m_cam_param_rgb;
        return true;
    }
    return false;
}

void DatasetLoader_Tele::Reset() {
    m_cursor = 0;
    frame_index = m_frames.empty() ? 0 : m_frames.front().frame_number;
}

bool DatasetLoader_Tele::Retrieve() {
    if (m_cursor >= m_frames.size()) return false;
    const auto &frame = m_frames.at(m_cursor++);

    m_rgb = cv::imread(frame.rgb, cv::IMREAD_COLOR);
    if (m_rgb.empty()) {
        throw std::runtime_error("Unable to read Tele RGB frame: " + frame.rgb);
    }

    m_d = cv::imread(frame.depth, cv::IMREAD_UNCHANGED);
    if (m_d.empty()) {
        throw std::runtime_error("Unable to read Tele depth frame: " + frame.depth);
    }
    if (m_d.channels() > 1) cv::cvtColor(m_d, m_d, cv::COLOR_BGR2GRAY);
    if (m_d.type() != CV_16UC1) {
        cv::Mat converted;
        m_d.convertTo(converted, CV_16U);
        m_d = converted;
    }
    for (int r = 0; r < m_d.rows; ++r) {
        for (int c = 0; c < m_d.cols; ++c) {
            auto &value = m_d.at<unsigned short>(r, c);
            if (value > static_cast<unsigned short>(m_dataset->max_depth)) value = 0;
        }
    }

    LoadPoseJson(m_pose, frame.pose);
    LoadNetworkPoseJson(m_network_pose, frame.pose);
    frame_index = frame.frame_number;
    return true;
}

void DatasetLoader_Tele::LoadPoseJson(Eigen::Matrix4f &pose, const std::string &path) const {
    pose.setIdentity();
#ifdef COMPILE_WITH_JSON
    const auto json = ORUtils::JsonUtils::LoadJson(path);
    const auto &position = json["position"].array_items();
    const auto &quat = json["rotation_quat"].array_items();
    if (position.size() != 3 || quat.size() != 4) {
        throw std::runtime_error("Invalid Tele pose json: " + path);
    }

    // Tele pose json stores quaternions as [w, x, y, z].
    float qw = static_cast<float>(quat.at(0).number_value());
    float qx = static_cast<float>(quat.at(1).number_value());
    float qy = static_cast<float>(quat.at(2).number_value());
    float qz = static_cast<float>(quat.at(3).number_value());
    const float norm = std::sqrt(qx * qx + qy * qy + qz * qz + qw * qw);
    if (norm > 1e-8f) {
        qx /= norm;
        qy /= norm;
        qz /= norm;
        qw /= norm;
    }

    pose(0, 0) = 1.0f - 2.0f * (qy * qy + qz * qz);
    pose(0, 1) = 2.0f * (qx * qy - qz * qw);
    pose(0, 2) = 2.0f * (qx * qz + qy * qw);
    pose(1, 0) = 2.0f * (qx * qy + qz * qw);
    pose(1, 1) = 1.0f - 2.0f * (qx * qx + qz * qz);
    pose(1, 2) = 2.0f * (qy * qz - qx * qw);
    pose(2, 0) = 2.0f * (qx * qz - qy * qw);
    pose(2, 1) = 2.0f * (qy * qz + qx * qw);
    pose(2, 2) = 1.0f - 2.0f * (qx * qx + qy * qy);

    // Tele/Habitat poses use the OpenGL camera convention: +X right, +Y up,
    // and the camera looks along -Z. InSeg builds point clouds in the OpenCV
    // convention: +X right, +Y down, +Z forward. Convert camera coordinates
    // before handing the camera-to-world pose to the reconstruction pipeline.
    pose.block<3, 1>(0, 1) *= -1.0f;
    pose.block<3, 1>(0, 2) *= -1.0f;

    pose(0, 3) = static_cast<float>(position.at(0).number_value()) * 1000.0f;
    pose(1, 3) = static_cast<float>(position.at(1).number_value()) * 1000.0f;
    pose(2, 3) = static_cast<float>(position.at(2).number_value()) * 1000.0f;
#else
    throw std::runtime_error("Tele pose loading requires COMPILE_WITH_JSON.");
#endif
}

void DatasetLoader_Tele::LoadNetworkPoseJson(Eigen::Matrix4f &pose, const std::string &path) const {
    pose.setIdentity();
#ifdef COMPILE_WITH_JSON
    const auto json = ORUtils::JsonUtils::LoadJson(path);
    const auto &position = json["position"].array_items();
    const auto &quat = json["rotation_quat"].array_items();
    if (position.size() != 3 || quat.size() != 4) {
        throw std::runtime_error("Invalid Tele pose json: " + path);
    }

    // Legacy network-input compatibility for models trained before the
    // fixed-quaternion dataloader. Those models interpreted rotation_quat as
    // [x, y, z, w] and did not apply the Habitat/OpenGL -> OpenCV basis flip.
    float qx = static_cast<float>(quat.at(0).number_value());
    float qy = static_cast<float>(quat.at(1).number_value());
    float qz = static_cast<float>(quat.at(2).number_value());
    float qw = static_cast<float>(quat.at(3).number_value());
    const float norm = std::sqrt(qx * qx + qy * qy + qz * qz + qw * qw);
    if (norm > 1e-8f) {
        qx /= norm;
        qy /= norm;
        qz /= norm;
        qw /= norm;
    }

    pose(0, 0) = 1.0f - 2.0f * (qy * qy + qz * qz);
    pose(0, 1) = 2.0f * (qx * qy - qz * qw);
    pose(0, 2) = 2.0f * (qx * qz + qy * qw);
    pose(1, 0) = 2.0f * (qx * qy + qz * qw);
    pose(1, 1) = 1.0f - 2.0f * (qx * qx + qz * qz);
    pose(1, 2) = 2.0f * (qy * qz - qx * qw);
    pose(2, 0) = 2.0f * (qx * qz - qy * qw);
    pose(2, 1) = 2.0f * (qy * qz + qx * qw);
    pose(2, 2) = 1.0f - 2.0f * (qx * qx + qy * qy);

    pose(0, 3) = static_cast<float>(position.at(0).number_value()) * 1000.0f;
    pose(1, 3) = static_cast<float>(position.at(1).number_value()) * 1000.0f;
    pose(2, 3) = static_cast<float>(position.at(2).number_value()) * 1000.0f;
#else
    throw std::runtime_error("Tele pose loading requires COMPILE_WITH_JSON.");
#endif
}
