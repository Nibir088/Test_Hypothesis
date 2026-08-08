from setuptools import find_packages, setup

setup(
    name="test_hypothesis",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch",
        "torchvision",
        "numpy",
        "Pillow",
        "matplotlib",
        "opencv-python",
        "segment-anything",
        "timm",
    ],
    python_requires=">=3.9",
)
