# Define Diffcoal sources and headers

# --- Core library ---
set(${PROJECT_NAME}_CORE_SOURCES empty.cpp)

set(${PROJECT_NAME}_CORE_PUBLIC_HEADERS
    ${PROJECT_SOURCE_DIR}/include/diffcoal/collision/collision.hpp
    ${PROJECT_SOURCE_DIR}/include/diffcoal/collision/collision.hxx
    ${PROJECT_SOURCE_DIR}/include/diffcoal/collision/fwd.hpp
    ${PROJECT_SOURCE_DIR}/include/diffcoal/core/fwd.hpp
    ${PROJECT_SOURCE_DIR}/include/diffcoal/utils/helpers.hpp
    ${PROJECT_SOURCE_DIR}/include/diffcoal/utils/logger.hpp
    ${PROJECT_SOURCE_DIR}/include/diffcoal/utils/mesh_io.hpp
    ${PROJECT_SOURCE_DIR}/include/diffcoal/utils/mesh_io.hxx
    ${PROJECT_SOURCE_DIR}/include/diffcoal/utils/openmp.hpp
    ${PROJECT_SOURCE_DIR}/include/diffcoal/fwd.hpp)

set(_binary_headers_root ${${PROJECT_NAME}_BINARY_DIR}/include/diffcoal)
set(${PROJECT_NAME}_CORE_GENERATED_PUBLIC_HEADERS
    ${_binary_headers_root}/config.hpp 
    ${_binary_headers_root}/deprecated.hpp
    ${_binary_headers_root}/warning.hpp)

# --- Template instantiation ---
set(${PROJECT_NAME}_TEMPLATE_INSTANTIATION_PUBLIC_HEADERS
    ${PROJECT_SOURCE_DIR}/include/diffcoal/collision/collision.txx)

set(${PROJECT_NAME}_TEMPLATE_INSTANTIATION_SOURCES
    ${PROJECT_SOURCE_DIR}/src/collision/collision.cpp)

# --- Python bindings ---
set(${PROJECT_NAME}_BINDINGS_PYTHON_PUBLIC_HEADERS
    ${PROJECT_SOURCE_DIR}/include/diffcoal/bindings/python/fwd.hpp
    ${PROJECT_SOURCE_DIR}/include/diffcoal/bindings/python/core/constraints-problem.hpp
    ${PROJECT_SOURCE_DIR}/include/diffcoal/bindings/python/core/simulator.hpp)

set(${PROJECT_NAME}_BINDINGS_PYTHON_SOURCES
    ${PROJECT_SOURCE_DIR}/bindings/python/core/expose-contact-frame.cpp
    ${PROJECT_SOURCE_DIR}/bindings/python/core/expose-constraints-problem.cpp
    ${PROJECT_SOURCE_DIR}/bindings/python/core/expose-simulator.cpp
    ${PROJECT_SOURCE_DIR}/bindings/python/module.cpp)