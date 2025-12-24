# Create header-only target All other target will depend on it.
add_library(${PROJECT_NAME}_header_only INTERFACE ${${PROJECT_NAME}_HEADERS})

if(INITIALIZE_WITH_NAN)
    target_compile_definitions(${PROJECT_NAME}_header_only INTERFACE EIGEN_INITIALIZE_MATRICES_BY_NAN)
endif()

if(CHECK_RUNTIME_MALLOC)
    target_compile_definitions(${PROJECT_NAME}_header_only INTERFACE DIFFCOAL_EIGEN_CHECK_MALLOC
                                                                    EIGEN_RUNTIME_NO_MALLOC)
endif(CHECK_RUNTIME_MALLOC)
if(ENABLE_TEMPLATE_INSTANTIATION)
    target_compile_definitions(${PROJECT_NAME}_header_only
                                INTERFACE DIFFCOAL_ENABLE_TEMPLATE_INSTANTIATION)
endif()


target_link_libraries(${PROJECT_NAME}_header_only INTERFACE Eigen3::Eigen coal::coal OpenMP::OpenMP_CXX
                                                            Boost::boost Boost::serialization)

# add precompiled header for eigen headers
if(USE_PRECOMPILED_HEADERS)
    target_precompile_headers(${PROJECT_NAME}_header_only INTERFACE
                                ${PROJECT_SOURCE_DIR}/include/diffcoal/pch.hpp)
endif()

# Enable tracy profiling
if(DIFFCOAL_TRACY_ENABLE)
    modernize_target_link_libraries(
        ${PROJECT_NAME}_header_only
        SCOPE INTERFACE
        TARGETS Tracy::TracyClient)
endif()

target_include_directories(${PROJECT_NAME}_header_only INTERFACE $<INSTALL_INTERFACE:include>)

cxx_flags_by_compiler_frontend(MSVC "/bigobj" OUTPUT PUBLIC_OPTIONS FILTER)
target_compile_options(${PROJECT_NAME}_header_only INTERFACE ${PUBLIC_OPTIONS})
cxx_flags_by_compiler_frontend(MSVC "NOMINMAX" OUTPUT PUBLIC_DEFINITIONS)
target_compile_definitions(${PROJECT_NAME}_header_only INTERFACE ${PUBLIC_DEFINITIONS})

add_header_group(${PROJECT_NAME}_CORE_PUBLIC_HEADERS)
add_header_group(${PROJECT_NAME}_CORE_GENERATED_PUBLIC_HEADERS)

install(
    TARGETS ${PROJECT_NAME}_header_only
    EXPORT ${TARGETS_EXPORT_NAME}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

add_library(${PROJECT_NAME} SHARED ${${PROJECT_NAME}_CORE_SOURCES})
target_link_libraries(${PROJECT_NAME} PUBLIC ${PROJECT_NAME}_header_only)

# Install main library
install(
    TARGETS ${PROJECT_NAME}
    EXPORT ${TARGETS_EXPORT_NAME}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)