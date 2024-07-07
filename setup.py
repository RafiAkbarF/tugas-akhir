from setuptools import setup, find_packages

setup(
    name="ocr_model",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "opencv-python-headless",
        "easyocr",
        "numpy",
        "matplotlib",
    ],
    entry_points={
        'console_scripts': [
            'ocr_model=ocr_model.ocr_functions:main',
        ],
    },
)
