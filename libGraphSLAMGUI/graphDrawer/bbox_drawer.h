#pragma once
#include <vector>
#include <memory>
#include <Eigen/Core>
#include "../../libGUI3D/libGUI3D/glShader.hpp"
#include "../../libGraphSLAM/graph.h"
#include "../../libGraphSLAM/node.h"

namespace PSLAM {

class BBoxDrawer {
public:
    BBoxDrawer() : bInited(false), mVertexCount(0), VAO(0), VBO(0) {}

    void Init() {
        if (bInited) return;
        mShader = std::make_unique<glUtil::Shader>();
        mShader->compileShader(vsShader, fsShader);
        glGenVertexArrays(1, &VAO);
        glGenBuffers(1, &VBO);
        bInited = true;
    }

    // Layout per node (32 verts total):
    //   [0..23]  bbox  — 12 edges × 2 verts
    //   [24..29] cross — 3 axes  × 2 verts
    //   [30..31] orient primary axis — 1 line × 2 verts
    void Update(Graph *g, int nodeFilterSize) {
        if (!g) return;
        std::vector<float> verts;
        const float cross_arm = 0.05f; // cross half-length in metres

        std::unique_lock<std::mutex> lock(g->mMutNode);
        for (const auto &kv : g->nodes) {
            const auto &n = kv.second;
            if (n->GetLabel() == Node::Unknown()) continue;
            if (n->surfels.empty()) continue;
            if (nodeFilterSize > 0 && (int)n->surfels.size() < nodeFilterSize) continue;

            Eigen::Vector3f lo, hi;
            Eigen::Vector3f ax1 = Eigen::Vector3f::UnitX();
            bool hasOrient = false;
            {
                std::unique_lock<std::mutex> sel_lock(n->mMutSelected);
                if (!n->mHasBBox) continue;
                lo = n->mBBox_min * 1e-3f;
                hi = n->mBBox_max * 1e-3f;
                ax1       = n->mOrientAxis1;
                hasOrient = n->mHasOrientation;
            }
            const Eigen::Vector3f c = (lo + hi) * 0.5f;

            auto addEdge = [&](float x0,float y0,float z0,float x1,float y1,float z1){
                verts.insert(verts.end(), {x0,y0,z0, x1,y1,z1});
            };
            auto addEdge3 = [&](const Eigen::Vector3f &a, const Eigen::Vector3f &b){
                addEdge(a.x(),a.y(),a.z(), b.x(),b.y(),b.z());
            };

            // bbox (24 verts)
            addEdge(lo.x(),lo.y(),lo.z(), hi.x(),lo.y(),lo.z());
            addEdge(hi.x(),lo.y(),lo.z(), hi.x(),hi.y(),lo.z());
            addEdge(hi.x(),hi.y(),lo.z(), lo.x(),hi.y(),lo.z());
            addEdge(lo.x(),hi.y(),lo.z(), lo.x(),lo.y(),lo.z());
            addEdge(lo.x(),lo.y(),hi.z(), hi.x(),lo.y(),hi.z());
            addEdge(hi.x(),lo.y(),hi.z(), hi.x(),hi.y(),hi.z());
            addEdge(hi.x(),hi.y(),hi.z(), lo.x(),hi.y(),hi.z());
            addEdge(lo.x(),hi.y(),hi.z(), lo.x(),lo.y(),hi.z());
            addEdge(lo.x(),lo.y(),lo.z(), lo.x(),lo.y(),hi.z());
            addEdge(hi.x(),lo.y(),lo.z(), hi.x(),lo.y(),hi.z());
            addEdge(hi.x(),hi.y(),lo.z(), hi.x(),hi.y(),hi.z());
            addEdge(lo.x(),hi.y(),lo.z(), lo.x(),hi.y(),hi.z());

            // cross (6 verts)
            addEdge(c.x()-cross_arm,c.y(),c.z(), c.x()+cross_arm,c.y(),c.z());
            addEdge(c.x(),c.y()-cross_arm,c.z(), c.x(),c.y()+cross_arm,c.z());
            addEdge(c.x(),c.y(),c.z()-cross_arm, c.x(),c.y(),c.z()+cross_arm);

            // orientation axes (4 verts): arm proportional to horizontal extent
            // ax1 (primary): full arm; ax2 (secondary): half arm
            if (hasOrient) {
                const float h_extent = std::max(hi.x()-lo.x(), hi.z()-lo.z());
                const float arm1 = std::max(0.1f, h_extent * 0.5f);
                addEdge3(c, c + arm1 * ax1); // primary axis: center → tip
            } else {
                addEdge3(c, c); // degenerate (invisible)
            }
        }

        mVertexCount = verts.size() / 3;
        if (mVertexCount == 0) return;

        glBindVertexArray(VAO);
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, verts.size() * sizeof(float), verts.data(), GL_DYNAMIC_DRAW);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), nullptr);
        glEnableVertexAttribArray(0);
        glBindVertexArray(0);
    }

    void Draw(const Eigen::Matrix4f &proj, const Eigen::Matrix4f &view,
              const Eigen::Vector4f &bboxColor   = {0.f, 0.8f, 1.f, 1.f},
              const Eigen::Vector4f &crossColor  = {1.f, 0.f,  0.f, 1.f},
              const Eigen::Vector4f &orientColor = {1.f, 0.5f, 0.f, 1.f})  // orange
    {
        if (!bInited || mVertexCount == 0) return;

        mShader->use();
        mShader->set("projection", proj);
        mShader->set("view", view);

        glBindVertexArray(VAO);

        // 32 verts per node: 24 bbox + 6 cross + 2 primary axis
        const int vertsPerNode = 32;
        const int nodes = (int)mVertexCount / vertsPerNode;
        for (int i = 0; i < nodes; ++i) {
            const int base = i * vertsPerNode;
            glLineWidth(1.5f);
            mShader->set("color", bboxColor);
            glDrawArrays(GL_LINES, base, 24);

            mShader->set("color", crossColor);
            glDrawArrays(GL_LINES, base + 24, 6);

            glLineWidth(3.f);
            mShader->set("color", orientColor);
            glDrawArrays(GL_LINES, base + 30, 2);
        }

        glLineWidth(1.f);
        glBindVertexArray(0);
    }

private:
    bool         bInited;
    size_t       mVertexCount;
    unsigned int VAO, VBO;
    std::unique_ptr<glUtil::Shader> mShader;

    const std::string vsShader =
        "#version 330 core\n"
        "layout(location=0) in vec3 aPos;\n"
        "uniform mat4 view;\n"
        "uniform mat4 projection;\n"
        "void main(){gl_Position = projection * view * vec4(aPos, 1.0);}";

    const std::string fsShader =
        "#version 330 core\n"
        "out vec4 FragColor;\n"
        "uniform vec4 color;\n"
        "void main(){FragColor = color;}";
};

} // namespace PSLAM
