cmake_minimum_required(VERSION 2.6)
Project (DynamicWalker)
#SET(OPENSIM_INSTALL_DIR $ENV{OPENSIM_HOME} CACHE PATH "Path to OpenSim
#Install")
SET(OPENSIM_INSTALL_DIR /usr/local/OpenSim CACHE PATH "Path to OpenSim
Install")
SET(CMAKE_CONFIGURATION_TYPES "RelWithDebInfo;Release" CACHE STRING "Semicolon
separated list of supported cofiguration types, only supports Release and
RelWithDebInfo, anything else will be ignored." FORCE)
SET(TARGET DynamicWalkerBuild CACHE STRING "The name of the executable")
SET(SOURCE DynamicWalkerBuildModel.cpp CACHE STRING "The name of the soure
files")

SET(SIMTK_HEADERS_DIR ${OPENSIM_INSTALL_DIR}/sdk/include/SimTK/include)
SET(OPENSIM_HEADERS_DIR ${OPENSIM_INSTALL_DIR}/sdk/include)

INCLUDE_DIRECTORIES(${SIMTK_HEADERS_DIR} ${OPENSIM_HEADERS_DIR})

SET(OPENSIM_LIBS_DIR ${OPENSIM_INSTALL_DIR}/sdk/lib ${OPENSIM_INSTALL_DIR}/lib)
SET(OPENSIM_DLLS_DIR ${OPENSIM_INSTALL_DIR}/bin)

LINK_DIRECTORIES(${OPENSIM_LIBS_DIR} ${OPENSIM_DLLS_DIR})

ADD_EXECUTABLE(${TARGET} ${SOURCE})

SET(NameSpace "" CACHE STRING "Prefix for SimTK lib names, includes
trailng '_'. Leave empty to use stock SimTK libraries.")
MARK_AS_ADVANCED(NameSpace)
TARGET_LINK_LIBRARIES(${TARGET}
	debug osimSimulation_d optimized osimSimulation
	debug osimActuators_d optimized osimActuators
	debug osimCommon_d optimized osimCommon
	debug osimAnalyses_d optimized osimAnalyses
	debug osimTools_d optimized osimTools
	debug ${NameSpace}SimTKcommon_d optimized ${NameSpace}SimTKcommon
	debug ${NameSpace}SimTKmath_d optimized  ${NameSpace}SimTKmath
	debug ${NameSpace}SimTKsimbody_d optimized ${NameSpace}SimTKsimbody
	#SimTKlapack
	${PLATFORM_LIBS}
)
IF(WIN32)
	SET(PLATFORM_LIBS pthreadVC2)
ELSE(WIN32)
	SET(NameSpace "")
	IF(APPLE)
		SET(PLATFORM_LIBS SimTKAtlas)
	ELSE(APPLE)
		SET(PLATFORM_LIBS SimTKAtlas_Lin_generic)
	ENDIF(APPLE)
ENDIF(WIN32)
MARK_AS_ADVANCED(CMAKE_INSTALL_PREFIX)
