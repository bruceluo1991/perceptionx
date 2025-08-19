###############################################################################
# Bazel now uses Bzlmod by default to manage external dependencies.
# Please consider migrating your external dependencies from WORKSPACE to MODULE.bazel.
#
# For more details, please check https://github.com/bazelbuild/bazel/issues/18958
###############################################################################
module(
    name = "x.perception",
    version = "0.0.1",
)

bazel_dep(name = "rules_python", version = "1.2.0")
bazel_dep(name = "rules_uv", version = "0.63.0")


PYTHON_VERSION = "3.12"
PIP_INDEX_URL = "https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple/"


python = use_extension("@rules_python//python/extensions:python.bzl", "python")
python.toolchain(
    is_default = True,
    python_version = PYTHON_VERSION,
)

pip = use_extension("@rules_python//python/extensions:pip.bzl", "pip")
pip.parse(
    environment = {
        "PIP_INDEX_URL": PIP_INDEX_URL,
    },
    hub_name = "pip",
    python_version = PYTHON_VERSION,
    requirements_lock = "//third_party/python:requirements_lock.txt",
)
use_repo(pip, "pip")

bazel_dep(name = "aspect_rules_py", version = "1.3.2")
