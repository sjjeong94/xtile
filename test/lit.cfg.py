# -*- Python -*-

import os

import lit.formats

from lit.llvm import llvm_config

config.name = "XT"
config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)
config.suffixes = [".mlir"]
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.xt_obj_root, "test")

llvm_config.with_system_environment(["HOME", "INCLUDE", "LIB", "TMP", "TEMP"])
llvm_config.use_default_substitutions()

config.excludes = ["Inputs", "CMakeLists.txt", "README.txt", "LICENSE.txt"]
config.xt_tools_dir = os.path.join(config.xt_obj_root, "bin")

tool_dirs = [config.xt_tools_dir, config.llvm_tools_dir]
tools = ["xt-opt", "mlir-opt", "FileCheck", "not", "count"]
llvm_config.add_tool_substitutions(tools, tool_dirs)
