from glob import glob
import os

from setuptools import setup

package_name = "followme_ros"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
        (os.path.join("share", package_name, "rviz"), glob("rviz/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Vedant Parnaik",
    maintainer_email="vedantparnaik@gmail.com",
    description="ROS 2 wrapper for the followme brain and the simkit world.",
    license="AGPL-3.0-only",
    entry_points={
        "console_scripts": [
            "world = followme_ros.world_node:main",
            "brain = followme_ros.brain_node:main",
        ],
    },
)
