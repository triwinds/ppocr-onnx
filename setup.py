import setuptools

with open("README.md", "r", encoding='utf-8') as fh:
    long_description = fh.read()

with open('requirements.txt', 'r', encoding='utf-8') as f:
    requirements = f.readlines()

setuptools.setup(
    name="ppocr-onnx",
    version="0.0.3.2",
    author="triwinds",
    author_email="triwinds@foxmail.com",
    license='Apache 2.0',
    description="利用 onnxruntime 及 PaddleOCR 提供的模型, 对图片中的文字进行检测与识别.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/triwinds/ppocr-onnx",
    packages=setuptools.find_packages(),
    install_requires=requirements,
    # include_package_data=True,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
