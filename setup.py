from setuptools import setup, find_packages

setup(
    name="banner_layout_analyzer",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "opencv-python>=4.8.0",
        "ultralytics>=8.0.0",
        "numpy>=1.24.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "Pillow>=9.0.0",
        "matplotlib>=3.7.0",
        "datasets>=2.15.0",
        "huggingface-hub>=0.19.0",
        "pandas>=2.0.0"
    ]
) 