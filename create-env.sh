uv venv --python 3.10

# Check if linux/macos or windows
if [ "$(uname)" == "Linux" ] || [ "$(uname)" == "Darwin" ]; then
    source .venv/bin/activate
elif [[ "$(uname -s)" == *"MINGW64_NT"* || "$(uname -s)" == *"MSYS_NT"* ]]; then
    source .venv/Scripts/activate
else
    echo "Unsupported OS. Please use Linux/macOS or Windows."
    exit 1
fi


uv sync --reinstall

uv pip install ruff huggingface_hub[hf_xet]
uv pip install peft
uv add "setuptools<81"
