# MarigoldRS: Text-Guided Real-time Monocular Depth Estimation for Remote Sensing
![](assets/picture1.png)
## Introduction
MarigoldRS is an advanced monocular depth estimation model specifically tailored for remote sensing imagery. Building upon the robust foundation of the original Marigold model, MarigoldRS introduces a key innovation: the integration of textual prompts as an additional input. This allows the model to leverage semantic context to generate more accurate and coherent depth maps, especially in visually ambiguous scenes common in remote sensing.

This model is designed to work with our accompanying benchmark dataset, [RS3DBench](https://huggingface.co/datasets/RS3DBench/RS3DBench).

## Getting Started
### Training
To train the model from scratch or fine-tune it on your own data, use the following command:

```bash
python train.py --config your/config/file.yaml
```

To resume a previous training run from a checkpoint:

```bash
python train.py --resume_run path/to/your/run/ckpt
```
 
### Inference
To run inference on a directory of RGB images and evaluate them against ground truth DEMs, use the `run.py` script.

Example Command:
```bash
python run.py \
    --input_rgb_dir {PATH_TO_RGB_IMAGES} \
    --output_dir {PATH_TO_SAVE_OUTPUT} \
    --checkpoint {PATH_TO_MODEL_CHECKPOINT} \
    --test_dem_dir {PATH_TO_GROUND_TRUTH_DEMS} \
    --denoise_steps 50 \
    --processing_res 0
    --text_prompt_path (optional) {PATH_TO_TEXT_PROMPTS}

```

Argument Explanations:

- --input_rgb_dir: Path to the directory containing your input RGB images.

- --output_dir: Path to the directory where the generated depth maps will be saved.

- --checkpoint: Path to the pre-trained model checkpoint file or directory.

- --test_dem_dir: Path to the directory containing the ground truth DEMs for evaluation.

- --denoise_steps: Number of denoising steps for the diffusion process.

- --processing_res: Processing resolution. 0 means using the original image resolution.

- --text_prompt_path: (Optional) Path to a text files containing prompts corresponding to each input image.
## Dataset
This model was trained and evaluated on the RS3DBench dataset. This dataset provides 54,951 pairs of remote sensing images, pixel-aligned depth maps, and corresponding textual descriptions.

You can access the full dataset on [Hugging Face](https://huggingface.co/datasets/RS3DBench/RS3DBench).