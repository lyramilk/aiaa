cmake_minimum_required(VERSION 3.5)
cmake_policy(SET CMP0048 NEW)
#定义工程名及版本
project(cudabp VERSION 1.0.0 LANGUAGES CXX CUDA)
set(soversion 1.0)

# 包含公共代码*
include(${PROJECT_SOURCE_DIR}/libmilk.cmake)

# 设置环境变量*
libmilk_cmake_init_env()

# 搜索代码
unset(codes)
unset(headers)

aux_source_directory(src codes)
file(GLOB cuda_codes "src/*.cu")
list(APPEND codes ${cuda_codes})

file(GLOB headers RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} include/lyramilk/tensor/*.h)

#搜索子模块


#定义目标*
libmilk_cmake_def(${PROJECT_NAME} ${PROJECT_NAME}-shared cudabp codes headers ${PROJECT_NAME}-alone)
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} include/lyramilk/tensor)
if(${PROJECT_NAME}-shared)
else()
endif()


#定义主程序
add_executable(test1 demo/test1.cc)
install(TARGETS test1 DESTINATION bin)
#target_link_libraries(test1 PUBLIC -Wl,--whole-archive cudabp ${PROJECT_NAME} -Wl,--no-whole-archive ${LIBS_DEPENDS})
target_link_libraries(test1 PUBLIC cudabp ${PROJECT_NAME} ${LIBS_DEPENDS})
