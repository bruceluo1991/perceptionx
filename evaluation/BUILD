load("@aspect_rules_py//py:defs.bzl",  "py_library", "py_binary")
load("@rules_python//python:defs.bzl", "py_test")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "geometry_utils",
    srcs = ["geometry_utils.py"],
    deps = [
        "//common",
        "@pip//numpy"
    ],
)

py_test(
    name = "geometry_utils_test",
    srcs = ["geometry_utils_test.py"],
    deps = [
        ":geometry_utils",
        "@pip//pytest",
    ],
)

py_library(
    name = "evaluation",
    srcs = ["evaluation.py"],
    deps = [
        ":geometry_utils",
        "//common",
        "@pip//numpy",
    ],
)

py_test(
    name = "evaluation_test",
    srcs = ["evaluation_test.py"],
    deps = [
        ":evaluation",
        "//common",
        "@pip//pytest",
        "@pip//numpy",
    ],
)
