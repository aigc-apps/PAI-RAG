## 1. Clone Repo and checkout branch
```bash
git clone https://github.com/aigc-apps/PAI-RAG.git
cd PAI-RAG
git checkout experiments/aworld_gaia
```

## 2. 按照 AWorld 要求准备环境和GAIA数据集
### 2.1. Set Up Conda Environment

Create and activate a dedicated Conda environment for GAIA:

```bash
conda env create -f examples/gaia/aworld-gaia.yml
conda activate aworld-gaia
```

### 2.2. Install AWorld Framework

Install the AWorld framework:

```bash
# Install PDF processing dependencies
pip install "marker-pdf[full]" --no-deps

# Install AWorld
python setup.py install
```

**注意**

实际运行过程中发现`examples/gaia/aworld-gaia.yml`中要求的`surya-ocr==0.14.6`版本太低可能会出错，若启动`mcp__image`遇到报错，建议升级`pip install surya-ocr==0.15.4`。

### 2.3. Install MCP Tool Dependencies

#### Install Playwright
```bash
playwright install chromium --with-deps --no-shell
```

#### Install System Dependencies

**For macOS:**
```bash
brew install libmagic
brew install ffmpeg
brew install --cask libreoffice
```

> **Note**: Install Homebrew from [brew.sh](https://brew.sh/) if not already installed.

**For Linux:**
```bash
apt-get install -y --no-install-recommends libmagic1 libreoffice ffmpeg
```

### 2.4. Prepare GAIA Dataset

Download the GAIA dataset from Hugging Face:

```bash
git clone git@hf.co:datasets/gaia-benchmark/GAIA examples/gaia/GAIA
```

> **⚠️ Important**: 
> - You need to configure Hugging Face SSH keys to access the GAIA repository
> - The dataset path will be used as the `GAIA_DATASET_PATH` variable in your `.env` file

## 3. 配置环境变量

在PAI-RAG根目录下:

```bash
cp .env.template .env
```

Edit the `.env` file and replace all `{YOUR_CONFIG}` placeholders with your actual configuration values.

## 4. 运行 GAIA Agent

在PAI-RAG根目录下:

```bash
python examples/gaia/gaia_agent_runner_pairag.py --testfile gaia_level_1.jsonl
```

说明：
- `testfile` 参数接受一个JSONL的文件，格式必须和 `GAIA/2023/validation/metadata.jsonl` 保持一致，**首次运行建议只取1～3条进行E2E测试**。
- `examples/gaia/gaia_agent_runner_pairag.py` 是修改了 `examples/gaia/gaia_agent_runner.py` 文件，支持直接从JSONL文件读取query set运行，并保存结果。
- `gaia_level_1.jsonl` 是从 `examples/gaia/GAIA/2023/validation/metadata.jsonl` 中筛选了 level 为 1 的数据（共53条）。
- 每一次运行结果保存在 `examples/gaia/output` 目录下，有一个总体的结果汇总文件 `examples/gaia/output/results.jsonl`，以及对 `testfile` 里每一条query的运行日志。