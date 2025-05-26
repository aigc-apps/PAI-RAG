from setuptools import setup, find_packages

setup(
    name="pairag.file",  # 包名
    version="0.1.0",
    author="Your Name",
    author_email="you@example.com",
    description="PAI-RAG file utilities.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/aigc-apps/PAI-RAG",
    packages=find_packages(where="src/pairag/file"),  # 从 src/ 下查找包
    package_dir={"": "src/pairag"},  # 指定源码根目录
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
    install_requires=[
        "requests>=2.26.0",
        "click>=8.0",
        "llama-index-core==0.12.12",
        "llama-index-readers-file>=0.4.3",
        "docx2txt>=0.8",
        "pydantic>=2.7.0",
        "oss2>=2.18.5",
        "torch==2.2.2",
        "transformers==4.51.3",
        "openpyxl>=3.1.2",
        "xlrd>=2.0.1",
        "markdown>=3.6",
        "chardet>=5.2.0",
        "peft>=0.12.0",
        "python-pptx>=1.0.2",
        "aspose-slides>=24.10.0",
        "datasketch>=1.6.5",
        "mistletoe>=1.4.0",
        "html2text>=2024.2.26",
        "python-docx>=1.1.2",
        "numpy==1.26.4",
        "loguru>=0.7.3",
        "magic-pdf[full]==1.3.10",
    ],
)
