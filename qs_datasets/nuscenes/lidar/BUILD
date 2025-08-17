load("@aspect_rules_py//py:defs.bzl",  "py_binary")
load("@rules_python//python:defs.bzl", "py_test")

package(default_visibility = ["//visibility:public"])


py_binary(
    name = "frame",
    srcs = ["frame.py"],
    deps = [
        "@pip//numpy",
        "@pip//scipy",
        "//qs_datasets/nuscenes:pose",
    ],
)

py_binary(
    name = "scenes",
    srcs = ["scenes.py"],
    deps = [
        "@pip//numpy",
        "@pip//scipy",
        "//qs_datasets/nuscenes:pose",
        ":frame",
    ],
)
