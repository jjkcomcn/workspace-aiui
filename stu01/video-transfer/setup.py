from setuptools import setup, find_packages

setup(
    name="video_transfer",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'opencv-python',
        'torch',
        'tqdm'
    ],
    entry_points={
        'console_scripts': [
            'video-transfer=video_transfer.main:main',
            'camera-processor=video_transfer.scripts.camera_processor:main'
        ],
    },
)
