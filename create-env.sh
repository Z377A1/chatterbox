uv venv --python 3.10

source .venv/bin/activate

uv sync --reinstall

uv pip install ruff huggingface_hub[hf_xet]
uv pip install peft
uv add "setuptools<81"
