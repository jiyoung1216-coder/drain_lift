from setuptools import find_packages, setup
from glob import glob
import os
from pathlib import Path

package_name = 'grating_detector_py'

# 우리가 패키지 안에 둘 모델 파일
model_src_path = 'resource/weights/best.pt'

data_files = [
    # ROS2가 이 패키지를 찾을 수 있게 하는 메타데이터
    ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
    # package.xml
    ('share/' + package_name, ['package.xml']),
]

# 모델 파일이 있으면 설치 대상에 추가
if os.path.exists(model_src_path):
    data_files.append(
        (
            os.path.join('share', package_name, 'resource', 'weights'),
            [model_src_path],
        )
    )

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=data_files,
    install_requires=[
        'setuptools',
        # detector_node_rgb.py 에서 get_package_share_directory 쓸 거니까
        'ament_index_python',
    ],
    zip_safe=True,
    maintainer='sj-desktop',
    maintainer_email='sj-desktop@todo.todo',
    description='YOLO-based grating detector (Python)',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            # ros2 run grating_detector_py detector_rgb
            'detector_rgb = grating_detector_py.detector_node_rgb:main',
        ],
    },
)
