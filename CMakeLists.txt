cmake_minimum_required(VERSION 2.8)

if(POLICY CMP0020)
  cmake_policy(SET CMP0020 NEW)
endif()

### Apple specific setup
if(APPLE)
message(STATUS "+-+-+-+-+-+- APPLE -+-+-+-+-+-+")
set(CMAKE_CXX_COMPILER "/Users/dateng/Developer/apple_legacy/Xcode_7.2.xctoolchain/usr/bin/c++")
set(CMAKE_C_COMPILER "/Users/dateng/Developer/apple_legacy/Xcode_7.2.xctoolchain/usr/bin/cc")
endif()
### Windows specific setup
if(WIN32)
message(STATUS "+-+-+-+-+-+- WINDOWS -+-+-+-+-+-+")

endif()

project(testMatGUI)

###### Add CUDA support
## Find CUDA
add_definitions( -DCUDA_API_PER_THREAD_DEFAULT_STREAM )
find_package(CUDA QUIET REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Ofast")
# set( CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS}  -O3")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_21")

## Compile cuda code
cuda_add_library(
	cudaKernels
	"./cudaKernels.cu"
	)

###-------------------- Find ITK.------------------------------
find_package(ITK REQUIRED)
# FIND_PACKAGE( FFTW )
include(${ITK_USE_FILE})

###-------------------- Find VTK ------------------------------
find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

include_directories( "D:/RLSimulation/tools/dependencies/include/matio" )
include_directories( "D:/RLSimulation/tools/dependencies/include/hdf5" )
include_directories( "D:/RLSimulation/tools/dependencies/include/zlib" )
link_directories( "D:/RLSimulation/tools/dependencies/lib" )
# Personal Toolbox
include_directories( "D:/C++/Toolbox/include" )
link_directories( "D:/C++/toolbox/bin_vs2012/Release" )

###------------------- Configure QT ------------------------
# if(${VTK_VERSION} VERSION_GREATER "6" AND VTK_QT_VERSION VERSION_GREATER "4")
  # Instruct CMake to run moc automatically when needed.
  set(CMAKE_AUTOMOC ON)
  find_package(Qt5Widgets REQUIRED QUIET)
  message(STATUS "Qt5Widgets found")
if (WIN32)
  include_directories( "D:/RLSimulation/Qt-5.2.1/5.2.1/msvc2012_64_opengl/include" )
  include_directories( "D:/RLSimulation/Qt-5.2.1/5.2.1/msvc2012_64_opengl/include/QtCore" )
  include_directories( "D:/RLSimulation/Qt-5.2.1/5.2.1/msvc2012_64_opengl/include/win32-msvc2012" )
  include_directories( "D:/RLSimulation/Qt-5.2.1/5.2.1/msvc2012_64_opengl/include/QtGui" )
  include_directories( "D:/RLSimulation/Qt-5.2.1/5.2.1/msvc2012_64_opengl/include/QtWidgets" )
endif()
# else()
#  find_package(Qt4 REQUIRED)
#  include(${QT_USE_FILE})
#endif()

#------------------- Assuming using VTK of version greater than 6.0 and QT version greater than 4.0
file(GLOB UI_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.ui)
file(GLOB QT_WRAP ${CMAKE_CURRENT_SOURCE_DIR}/build/ui_*.h)
qt5_wrap_ui(UISrcs ${UI_FILES} )
message(STATUS "------------------ QT WRAP/ui files -----------------------------")
message(STATUS ${QT_WRAP})
message(STATUS ${UI_FILES})


include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})

add_executable(Main 
                ${CMAKE_CURRENT_SOURCE_DIR}/main.cxx
				${CMAKE_CURRENT_SOURCE_DIR}/RenderWindowUISingleInheritance.cxx
				${CMAKE_CURRENT_SOURCE_DIR}/RenderWIndowUISingleInheritance.h
				${CMAKE_CURRENT_SOURCE_DIR}/imageimportdialog.cpp
				${CMAKE_CURRENT_SOURCE_DIR}/imageimportdialog.h
                ${UISrcs}
                ${QT_WRAP}
				)
qt5_use_modules(Main Core Gui)
target_link_libraries(Main 
	libmatio.lib
	itk_io.lib
	cudaKernels
	${VTK_LIBRARIES}
	${ITK_LIBRARIES}
	)