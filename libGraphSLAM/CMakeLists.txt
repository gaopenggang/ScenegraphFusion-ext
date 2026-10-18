SET(targetname libGraphSLAM)

INCLUDE(UseEigen3)
INCLUDE(UseOpenCV)

if(APPLE)
    set(INSEG_LIB ${PROJECT_SOURCE_DIR}/lib/libInSegLib_osx.a)
else()
    set(INSEG_LIB ${PROJECT_SOURCE_DIR}/lib/libInSegLib_linux.a)
endif()

FILE(GLOB sources "*.cpp")
FILE(GLOB headers "*.h")
IF(BUILD_GRAPHPRED)
    INCLUDE(UseOnnxRuntime)
    FILE(GLOB src_gp "graphPredictor/*.cpp")
    FILE(GLOB h_gp   "graphPredictor/*.h")
    SET(sources ${sources} ${src_gp})
    SET(headers ${headers} ${h_gp})
ENDIF(BUILD_GRAPHPRED)

IF(BUILD_ROS_PUBLISH)
    FILE(GLOB src_ros "rosPublisher/*.cpp")
    FILE(GLOB h_ros   "rosPublisher/*.h")
    SET(sources ${sources} ${src_ros})
    SET(headers ${headers} ${h_ros})
ENDIF(BUILD_ROS_PUBLISH)



ADD_LIBRARY(${targetname} ${sources} ${headers})
TARGET_LINK_LIBRARIES(${targetname}
        PUBLIC  ${INSEG_LIB}
)
TARGET_INCLUDE_DIRECTORIES(${targetname}
        PUBLIC ../lib)
IF(BUILD_GRAPHPRED)
    INCLUDE(LinkOnnxRuntime)
    target_compile_definitions(${targetname} PUBLIC COMPILE_WITH_GRAPHPRED)
ENDIF(BUILD_GRAPHPRED)
IF(BUILD_ROS_PUBLISH)
    find_package(ament_cmake REQUIRED)
    find_package(rclcpp REQUIRED)
    find_package(std_msgs REQUIRED)
    find_package(geometry_msgs REQUIRED)
    find_package(scene_msgs REQUIRED)
    ament_target_dependencies(${targetname} PUBLIC rclcpp std_msgs geometry_msgs scene_msgs)
    target_compile_definitions(${targetname} PUBLIC COMPILE_WITH_ROS_PUBLISH)
ENDIF(BUILD_ROS_PUBLISH)
INCLUDE(Linktinyply)
INCLUDE(LinkJson)
