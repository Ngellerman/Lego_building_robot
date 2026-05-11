from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'tuck_high'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ee106a-acz',
    maintainer_email='ziyuanxu2333@gmail.com',
    description='Move UR7e to high observation tuck pose.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'tuck_high = tuck_high.tuck_high:main',
        ],
    },
)
