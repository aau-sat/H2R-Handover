# Segmentation with SAM3

## Installation

Follow SAM3 installation instructions [here](./sam3/README.md#Installation).

Then, install the `transformers` library from source:

```bash
conda activate sam3
cd sam3
git clone https://github.com/huggingface/transformers.git
cd transformers
pip install '.[torch]'
```

## Offline use (recommended)

The service is configured to **always run offline**. On the first run, it will automatically download and cache the SAM3 model locally. Subsequent runs will use the cached model without any internet access.

**First run** (requires internet to download model ~900MB):
```bash
python segmentation_service_multiprompt.py
```

**Subsequent runs** (fully offline):
```bash
python segmentation_service_multiprompt.py
```

The model is cached in `./sam3_model_cache/` by default. To use a custom cache location:
```bash
export SAM3_MODEL_PATH=/your/custom/cache/path
python segmentation_service_multiprompt.py
```

Alternatively, you can download the model separately with:
```bash
python download_sam3_model.py [optional_custom_cache_dir]
```

## Running multiple text prompts
Activate your environment and run the following command:

```bash
python test_multiple_promts.py --input <path_to_image>
```

You can also specify your own text prompts by adding the `--prompts` argument, and also output filename.
The mask will be saved at the same path as the input image.