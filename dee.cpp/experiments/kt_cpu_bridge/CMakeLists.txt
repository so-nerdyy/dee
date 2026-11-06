cmake_minimum_required(VERSION 3.16)
project(kt_cpu_bridge CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Standalone prototype build. Intentionally NOT added to dee.cpp root build.
add_library(kt_bridge STATIC
  src/reference_cpu_executor.cpp
  src/kt_cpu_executor.cpp)
target_include_directories(kt_bridge PUBLIC include)
if (MSVC)
  target_compile_options(kt_bridge PRIVATE /W4 /EHsc)
else()
  target_compile_options(kt_bridge PRIVATE -Wall -Wextra -O2)
endif()
add_executable(kt_bridge_smoke src/smoke_main.cpp)
target_link_libraries(kt_bridge_smoke PRIVATE kt_bridge)
