# ece285-project-lora-stablediffusion
“Course project for ECE 285: Fine-tuning Stable Diffusion with LoRA on chest X-ray datasets.”

# Fine-Tuning Stable Diffusion for Chest X-ray Synthesis

## Project Objectives
- Fine-tune a text-conditional latent diffusion model (LDM) to synthesize high-fidelity chest X-ray images controlled by domain-specific text prompts.
- Evaluate the quality, diversity, and realism of the generated synthetic images using established metrics such as:
  - **FID (Fréchet Inception Distance)**
  - **IS (Inception Score)**
- Demonstrate the feasibility and value of domain-adapted generative models in addressing data scarcity and accelerating AI development in medical imaging.

## Project Background
Medical imaging often suffers from limited access to well-annotated and diverse datasets due to:
- The need for expert radiological interpretation
- Strict privacy regulations
- Data isolation across healthcare institutions

Limited data availability, especially for rare pathologies or nuanced clinical presentations, poses challenges in training reliable AI systems for clinical use.  

**Latent Diffusion Models (LDMs)** generate high-fidelity images and are particularly useful in data-scarce domains. However, models trained on natural images often fail to capture the specialized language and visuals of medical images.  

**Fine-tuning LDMs with medical image-text pairs allows:**
- Generation of synthetic images for rare or underrepresented cases
- Enhanced model robustness
- Scalable, safe, and efficient augmentation of medical datasets

## Model Architecture
This project leverages **Stable Diffusion 1.5 (SD1.5)** as the generative foundation. The pipeline consists of:

1. **Variational Autoencoder (VAE)**
2. **Denoising U-Net Backbone**
3. **Text Conditioning via CLIP**
4. **Low-Rank Adaptation (LoRA)**

## Dataset
**Source:** [Indiana University Chest X-rays on Kaggle](https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university)  

- **Images:** 7,466 chest X-ray images (Frontal & Lateral views) from 3,851 patients  
- **Text Prompts:** Extracted from the *impression* section of radiology reports, focusing on relevant findings.  
  - Excluded empty samples and those exceeding 77 tokens (CLIP tokenizer limit)
  - Total usable samples: 7,281 unique image-text pairs  
- **Split:**  
  - Training: 5,825 pairs (80%)  
  - Testing: 1,456 pairs (20%)  
- **Preprocessing:** Images resized to **128×128 pixels** due to memory limitations

## Fine-Tuning Experiment
- **Hardware:** NVIDIA P100 GPU (via Kaggle)  
- **Method:** LoRA applied exclusively to the U-Net component; all other components frozen  
- **Batch Size:** 4  
- **Implementation:**  
  - Model weights: [stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5)  
  - Libraries: `diffusers` v0.32.2, Hugging Face LoRA framework  

**Training Procedure:**
1. For each image-text pair \(i\) and timestep \(t\), sample Gaussian noise \(N_{i,t}\) in latent space `(h, w)`  
2. Add noise to the latent representation of the image: \(y_\text{pixel} \oplus T N_{i,t}\)  
3. U-Net \(\psi\) processes the noisy latent representation and encoded prompt \(E_T(x_\text{text})\) to predict the noise \(\hat{N}_{i,T}\)  
4. Compute **Mean Squared Error (MSE)** between true \(N_{i,t}\) and predicted noise \(\hat{N}_{i,T}\)  
5. Update weights of unfrozen components via backpropagation  

---

## References
- Stable Diffusion: [https://github.com/CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion)  
- Hugging Face Diffusers: [https://huggingface.co/docs/diffusers/index](https://huggingface.co/docs/diffusers/index)  
- Indiana Chest X-rays: [https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university](https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university)
