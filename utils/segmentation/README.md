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

## Running multiple text prompts
Activate your environment and run the following command:

```bash
python test_multiple_promts.py --input <path_to_image>
```

You can also specify your own text prompts by adding the `--prompts` argument, and also output filename.
The mask will be saved at the same path as the input image.