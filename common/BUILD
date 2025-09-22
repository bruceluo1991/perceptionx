load("@aspect_rules_py//py:defs.bzl",  "py_library", "py_binary")
load("@rules_python//python:defs.bzl", "py_test")
package(default_visibility = ["//visibility:public"])

py_library(
    name="common",
    srcs=["common.py"],
    deps=[
        "@pip//numpy",
    ]
)

py_test(
    name="common_test",
    srcs=["common_test.py"],
    deps=[
        ":common",
        "@pip//pytest",
    ],
)
