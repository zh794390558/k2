# https://cmake.org/cmake/help/latest/module/FindPython3.html#module:FindPython3
find_package(Python3 COMPONENTS Interpreter Development)
# find_package(pybind11 CONFIG)

if(Python3_FOUND)
    message(STATUS "Python3_FOUND = ${Python3_FOUND}")
    message(STATUS "Python3_EXECUTABLE = ${Python3_EXECUTABLE}")
    message(STATUS "Python3_LIBRARIES = ${Python3_LIBRARIES}")
    message(STATUS "Python3_INCLUDE_DIRS = ${Python3_INCLUDE_DIRS}")
    message(STATUS "Python3_LINK_OPTIONS = ${Python3_LINK_OPTIONS}")
    set(PYTHON_LIBRARIES ${Python3_LIBRARIES} CACHE STRING "python lib" FORCE)
    set(PYTHON_INCLUDE_DIR ${Python3_INCLUDE_DIRS} CACHE STRING "python inc" FORCE)
endif()

message(STATUS "PYTHON_LIBRARIES = ${PYTHON_LIBRARIES}")
message(STATUS "PYTHON_INCLUDE_DIR = ${PYTHON_INCLUDE_DIR}")
include_directories(${PYTHON_INCLUDE_DIR})

# if(pybind11_FOUND)
#     message(STATUS "pybind11_INCLUDES = ${pybind11_INCLUDE_DIRS}")
#     message(STATUS "pybind11_LIBRARIES=${pybind11_LIBRARIES}")
#     message(STATUS "pybind11_DEFINITIONS=${pybind11_DEFINITIONS}")
# endif()


# paddle libpaddle.so
# paddle include and link option
# -L/workspace/DeepSpeech-2.x/engine/venv/lib/python3.7/site-packages/paddle/libs -L/workspace/DeepSpeech-2.x/speechx/venv/lib/python3.7/site-packages/paddle/fluid -l:libpaddle.so -l:libdnnl.so.2 -l:libiomp5.so
set(EXECUTE_COMMAND "import os"
    "import paddle"
    "include_dir = paddle.sysconfig.get_include()"
    "paddle_dir=os.path.split(include_dir)[0]"
    "libs_dir=os.path.join(paddle_dir, 'libs')"
    "fluid_dir=os.path.join(paddle_dir, 'fluid')"
    "out=' '.join([\"-L\" + libs_dir, \"-L\" + fluid_dir])"
    "out += \" -l:libpaddle.so -l:libdnnl.so.2 -l:libiomp5.so\"; print(out)"
)
execute_process(
    COMMAND python -c "${EXECUTE_COMMAND}"
    OUTPUT_VARIABLE PADDLE_LINK_FLAGS
    RESULT_VARIABLE SUCESS)

message(STATUS PADDLE_LINK_FLAGS= ${PADDLE_LINK_FLAGS})
string(STRIP ${PADDLE_LINK_FLAGS} PADDLE_LINK_FLAGS)

# paddle compile option
# -I/workspace/DeepSpeech-2.x/engine/venv/lib/python3.7/site-packages/paddle/include
set(EXECUTE_COMMAND "import paddle"
    "include_dir = paddle.sysconfig.get_include()"
    "print(f\"-I{include_dir}\")"
)
execute_process(
    COMMAND python -c "${EXECUTE_COMMAND}"
    OUTPUT_VARIABLE PADDLE_COMPILE_FLAGS)
message(STATUS PADDLE_COMPILE_FLAGS= ${PADDLE_COMPILE_FLAGS})
string(STRIP ${PADDLE_COMPILE_FLAGS} PADDLE_COMPILE_FLAGS)

# for LD_LIBRARY_PATH
# set(PADDLE_LIB_DIRS /workspace/DeepSpeech-2.x/tools/venv/lib/python3.7/site-packages/paddle/fluid:/workspace/DeepSpeech-2.x/tools/venv/lib/python3.7/site-packages/paddle/libs/)
set(EXECUTE_COMMAND "import os"
    "import paddle"
    "include_dir=paddle.sysconfig.get_include()"
    "paddle_dir=os.path.split(include_dir)[0]"
    "libs_dir=os.path.join(paddle_dir, 'libs')"
    "fluid_dir=os.path.join(paddle_dir, 'fluid')"
    "out=':'.join([libs_dir, fluid_dir]); print(out)"
    )
execute_process(
    COMMAND python -c "${EXECUTE_COMMAND}"
    OUTPUT_VARIABLE PADDLE_LIB_DIRS)
message(STATUS PADDLE_LIB_DIRS= ${PADDLE_LIB_DIRS})