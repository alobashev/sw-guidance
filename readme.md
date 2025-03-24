# [CVPR 2025] Color Conditional Generation with Sliced Wasserstein Guidance

This repository contains the source code and instructions for reproducing the results presented in our CVPR 2025 paper "Color Conditional Generation with Sliced Wasserstein Guidance" by Alexander Lobashev, Maria Larchenko and Dmitry Guskov. SW-Guidance modifies the image generation process of diffusion models by incorporating the color distribution of a reference image directly into the sampling mechanism, thereby overcoming the limitations of sequential text-to-image and style-transfer approaches.

<!-- <img src="figures/main_image.png" width="100%">   -->

## Abstract

We propose SW-Guidance, a training-free approach for image generation conditioned on the color distribution of a reference image. While it is possible to generate an image with fixed colors by first creating an image from a text prompt and then applying a color style transfer method, this approach often results in semantically meaningless colors in the generated image. Our method solves this problem by modifying the sampling process of a diffusion model to incorporate the differentiable Sliced 1-Wasserstein distance between the color distribution of the generated image and the reference palette. Our method outperforms state-of-the-art techniques for color-conditional generation in terms of color similarity to the reference, producing images that not only match the reference colors but also maintain semantic coherence with the original text prompt.

## Features

- **Training-Free Approach:** No extra training is required.
- **Color-Conditional Generation:** Directly conditions image generation on a reference color palette.
- **Enhanced Sampling:** Incorporates Sliced 1-Wasserstein distance to align the generated image’s colors with the reference.

<img src="figures/main_scheme_short.png" width="100%">  
<img src="figures/intro_image_with_prompts.png" width="100%">  

## Compatibility with ControlNets  

Our method is compatible with ControlNets, allowing for even more precise control over image generation while maintaining color consistency.

<p align="center">  
  <img src="figures/newrebutal_1.png" width="100%">  
  <img src="figures/newrebutal_2.png" width="100%">  
  <img src="figures/newrebutal_4.png" width="100%">  
</p>  

## Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/your_username/sw-guidance.git
cd sw-guidance
pip install -r requirements.txt
```

## Usage

To generate an image, run the following command:

```bash
python generate.py --prompt "A description of your image" --reference_path "path/to/reference.jpg"
```

- Replace `"A description of your image"` with your desired text prompt.
- Replace `"path/to/reference.jpg"` with the path to your reference image.

Additional parameters and options are available. Please refer to the [Usage Guide](docs/usage.md) for more details.

## Results

You can find some sample outputs and comparisons with state-of-the-art techniques in the [Results](docs/results.md) section

## Citation

If you use our work, please cite the paper as follows:

```

```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
