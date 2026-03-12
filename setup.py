from __future__ import annotations

from packaging import tags
from setuptools import setup
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel


class BinaryWheel(_bdist_wheel):
    def finalize_options(self) -> None:
        super().finalize_options()
        self.root_is_pure = False

    def get_tag(self) -> tuple[str, str, str]:
        tag = next(tags.sys_tags())
        return tag.interpreter, tag.abi, tag.platform


setup(cmdclass={"bdist_wheel": BinaryWheel})
