#pragma once
#include "dataset3RScan.h"
#include "datasetScanNet.h"
#include "datasetTele.h"
#include "dataloader_3rscan.h"
#include "dataloader_scannet.h"
#include "dataloader_tele.h"
#include <sys/stat.h>

namespace PSLAM {
    struct  DataLoaderFactory {
        static bool IsFolder(const std::string &path) {
            struct stat sb{};
            return stat(path.c_str(), &sb) == 0 && S_ISDIR(sb.st_mode);
        }

        static bool LooksLikeTeleDataset(std::string path) {
            if (!path.empty() && path.back() == '/') path.pop_back();
            return IsFolder(path + "/rgb") && IsFolder(path + "/depth") && IsFolder(path + "/pose");
        }

        static DatasetLoader_base *Make(
                const std::string &pth, INPUTE_TYPE inputeType = DATASET_DETECT) {
            DatasetLoader_base *output = nullptr;
            // detect datatype by checking file name

            if(inputeType == DATASET_DETECT){
                std::cerr << "detect data type: ";
                if (LooksLikeTeleDataset(pth)) {
                    inputeType = DATASET_TELE;
                    std::cerr << "Tele";
                } else if(pth.find(".sens") != std::string::npos || pth.find("scene") != std::string::npos) {
                    inputeType = DATASET_SCANNET;
                    std::cerr << "ScanNet";
                } else {
                    inputeType = DATASET_3RSCAN;
                    std::cerr << "3RScan";
                }
            }

            // Create dataloader
            switch (inputeType) {
                case DATASET_3RSCAN: {
                    auto path = pth.back() == '/'? pth : pth+"/";
                    auto database = std::make_shared<Scan3RDataset>(inputeType, path);
                    output = new DatasetLoader_3RScan(database);
                    break;
                }
                case DATASET_SCANNET: {
                    auto database = std::make_shared<ScanNetDataset>(inputeType, pth);
                    output = new DatasetLoader_ScanNet(database);
                    break;
                }
                case DATASET_TELE: {
                    auto database = std::make_shared<TeleDataset>(inputeType, pth);
                    output = new DatasetLoader_Tele(database);
                    break;
                }
                case DATASET_DETECT:
                    break;

            }
            return output;
        }
    };
}
