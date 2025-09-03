load("@aspect_rules_py//py:defs.bzl",  "py_library", "py_binary")
load("@rules_python//python:defs.bzl", "py_test")

package(default_visibility = ["//visibility:public"])

py_test(
    name = "pose_test",
    srcs = [
        "pose_test.py",
    ],
    deps = [
      ":pose",
      "@pip//pytest",
    ],
)

py_library(
    name = "pose",
    srcs = ["pose.py"],
    deps = [
        "@pip//numpy",
        "@pip//scipy",
    ],
)

py_binary(
    name="dataset_generator",
    srcs=["dataset_generator.py"],
    deps=[
        ":pose",
        "@pip//numpy",
        "@pip//scipy",
    ]
)
