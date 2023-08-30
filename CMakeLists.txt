cmake_minimum_required(VERSION 3.20)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

project(VulkanExamples LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)


set(RESOURCE_INSTALL_DIR "" CACHE PATH "Path to install resources to (leave empty for running uninstalled)")


file(GLOB_RECURSE sources CONFIGURE_DEPENDS src/*.cpp)
file(GLOB_RECURSE headers CONFIGURE_DEPENDS base/*.h base/*.hpp)

# set the two variable with your won boost library
set(Boost_ROOT D:/boost)
set(BOOST_LIBRARY_DIRS D:/boost/lib)

find_package(Vulkan REQUIRED)
find_package(Threads REQUIRED)
find_package(Boost 1.83 REQUIRED)


add_definitions(-D_CRT_SECURE_NO_WARNINGS)
add_definitions(-DNDEBUG)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNOMINMAX -D_USE_MATH_DEFINES")

IF(MSVC)
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")
ENDIF(MSVC)


if(RESOURCE_INSTALL_DIR)
	add_definitions(-DVK_EXAMPLE_ASSETS_DIR=\"${RESOURCE_INSTALL_DIR}/\")
	add_definitions(-DVK_EXAMPLE_SHADERS_DIR=\"${RESOURCE_INSTALL_DIR}/shaders/\")
	install(DIRECTORY assets/ DESTINATION ${RESOURCE_INSTALL_DIR}/)
	install(DIRECTORY shaders/ DESTINATION ${RESOURCE_INSTALL_DIR}/shaders/)
else()
	add_definitions(-DVK_EXAMPLE_ASSETS_DIR=\"${CMAKE_SOURCE_DIR}/assets/\")
	add_definitions(-DVK_EXAMPLE_SHADERS_DIR=\"${CMAKE_SOURCE_DIR}/shaders/\")
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/")

add_executable(main ${sources} ${headers})

target_include_directories(main PUBLIC base)
target_include_directories(main PUBLIC vendor/stb)
target_include_directories(main PUBLIC vendor/glfw/include)
target_include_directories(main PUBLIC vendor/imgui)
target_include_directories(main PUBLIC vendor/tinyobjloader)
target_include_directories(main PUBLIC vendor/ktx/include)
target_include_directories(main PUBLIC vendor/ktx/utils)
target_include_directories(main PUBLIC vendor/ktx/other_include)
target_include_directories(main PUBLIC vendor/spdlog/include)

# vulkan
target_include_directories(main PRIVATE ${Vulkan_INCLUDE_DIR})
target_link_libraries(main ${Vulkan_LIBRARIES})

# glfw
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/vendor/glfw)
target_link_libraries(main glfw)

# spdlog
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/vendor/spdlog)
target_link_libraries(main spdlog::spdlog)

# multithread
target_link_libraries(main Threads::Threads)

# boost
target_include_directories(main PUBLIC ${Boost_INCLUDE_DIRS})
target_link_libraries(main ${BOOST_LIBRARY})

set(CMAKE_INSTALL_PREFIX ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})



# install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/assets/models/viking_room/ DESTINATION viking_room)
# install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/shaders/glsl/triangle/ DESTINATION shaders/triangle)


# # add_subdirectory(glad)
# # target_link_libraries(main PUBLIC glad)
