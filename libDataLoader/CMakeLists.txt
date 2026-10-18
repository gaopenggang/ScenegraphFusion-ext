SET(targetname libDataLoader)

INCLUDE(UseOpenCV)
INCLUDE(UseEigen3)

SET(sources
        src/dataloader_scannet.cpp
        src/dataloader_3rscan.cpp
        src/dataloader_tele.cpp
        ScanNetLoader/sensorData.cpp
)
SET(headers
        include/dataLoader/dataset_loader.h
        include/dataLoader/dataset_loader_facotry.h
        include/dataLoader/datasetTele.h
        include/dataLoader/dataloader_tele.h
)

ADD_LIBRARY(${targetname} ${sources} ${headers})
TARGET_INCLUDE_DIRECTORIES(${targetname} PUBLIC ./include)
