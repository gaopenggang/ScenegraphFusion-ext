#
# Try to find onnxruntime library and include path.
# Once done this will define
# OnnxRuntime_FOUND
# ONNXROOT_DIR
# OnnxRuntime_INCLUDE_DIRS
# OnnxRuntime_LIBRARY
#
# And will create an interface library
# OnnxRuntime

IF(ONNXROOT_DIR)
    SET(_OnnxRuntime_SEARCH_PATHS ${ONNXROOT_DIR})
ELSE()
    SET(_OnnxRuntime_SEARCH_PATHS /usr/local/)
ENDIF()

FIND_PATH(OnnxRuntime_INCLUDE_DIR
        NAMES onnxruntime_cxx_api.h
        PATHS ${_OnnxRuntime_SEARCH_PATHS}
        PATH_SUFFIXES include include/onnxruntime/core/session
        NO_DEFAULT_PATH
)
IF(OnnxRuntime_INCLUDE_DIR)
    IF(OnnxRuntime_INCLUDE_DIR MATCHES "/include/onnxruntime/core/session$")
        GET_FILENAME_COMPONENT(_ORT_DIR1 "${OnnxRuntime_INCLUDE_DIR}" DIRECTORY)
        GET_FILENAME_COMPONENT(_ORT_DIR2 "${_ORT_DIR1}" DIRECTORY)
        GET_FILENAME_COMPONENT(_ORT_DIR3 "${_ORT_DIR2}" DIRECTORY)
        GET_FILENAME_COMPONENT(_ORT_ROOT_DIR "${_ORT_DIR3}" DIRECTORY)
    ELSE()
        GET_FILENAME_COMPONENT(_ORT_ROOT_DIR "${OnnxRuntime_INCLUDE_DIR}" DIRECTORY)
    ENDIF()
    SET(ONNXROOT_DIR "${_ORT_ROOT_DIR}" CACHE PATH "Root directory of ONNX Runtime" FORCE)
    SET(OnnxRuntime_INCLUDE_DIRS ${OnnxRuntime_INCLUDE_DIR})
    IF(EXISTS "${ONNXROOT_DIR}/include")
        LIST(APPEND OnnxRuntime_INCLUDE_DIRS ${ONNXROOT_DIR}/include)
    ENDIF()
    IF(EXISTS "${ONNXROOT_DIR}/include/onnxruntime/core/session")
        LIST(APPEND OnnxRuntime_INCLUDE_DIRS ${ONNXROOT_DIR}/include/onnxruntime/core/session)
    ENDIF()

    FIND_LIBRARY(OnnxRuntime_LIBRARY
            NAMES
            onnxruntime
            libonnxruntime.dylib
            libonnxruntime.so
            PATHS
            ${ONNXROOT_DIR}/lib
            NO_DEFAULT_PATH
    )
ENDIF(OnnxRuntime_INCLUDE_DIR)

IF (ONNXROOT_DIR AND OnnxRuntime_LIBRARY)
    SET( OnnxRuntime_FOUND 1 CACHE STRING "Set to 1 if GLEW is found, 0 otherwise")
ELSE (ONNXROOT_DIR AND OnnxRuntime_LIBRARY)
    SET( OnnxRuntime_FOUND 0 CACHE STRING "Set to 1 if GLEW is found, 0 otherwise")
ENDIF (ONNXROOT_DIR AND OnnxRuntime_LIBRARY)

MARK_AS_ADVANCED( OnnxRuntime_FOUND )

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OnnxRuntime
        REQUIRED_VARS OnnxRuntime_LIBRARY OnnxRuntime_INCLUDE_DIRS)

if(OnnxRuntime_FOUND)
    if(NOT TARGET OnnxRuntime)
        add_library(OnnxRuntime UNKNOWN IMPORTED)
        set_target_properties(OnnxRuntime PROPERTIES
                INTERFACE_INCLUDE_DIRECTORIES "${OnnxRuntime_INCLUDE_DIRS}")
        set_target_properties(OnnxRuntime PROPERTIES
                IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
                IMPORTED_LOCATION "${OnnxRuntime_LIBRARY}")
        TARGET_COMPILE_DEFINITIONS(OnnxRuntime INTERFACE COMPILE_WITH_ONNX)
    endif()
endif()
