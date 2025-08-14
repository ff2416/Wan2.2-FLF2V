# Wan2.2-FLF2V
In this repository, we present Wan2.2-FLF2V, which builds upon WAN 2.2 Image-to-Video (I2V) and introduces several key enhancements:

- **Last Frame Constraint**: Enforces precise alignment between the generated last frame and the target frame, ensuring consistent video endpoints.

- **Bidirectional Denoising with Time Reversal Fusion**: Performs denoising both forward (first-to-last) and backward (last-to-first), fusing intermediate results at every step for superior temporal coherence. To accommodate bidirectional fusion, the original denoising formula—where each step depends on the previous one—has been redesigned, allowing non-continuous, step-wise integration of forward and backward denoised states.

- **Prompt-Adapted Temporal Attention**: During the reverse pass, temporal self-attention is rotated to align backward generation with the prompt, enabling bidirectionally refined, prompt-consistent video sequences.

With these improvements, we achieve First–Last–Frame-to-Video generation (FLF2V), enabling controllable and consistent video synthesis given the first and last frames as constraints.

## Run Wan2.2-FLF2V
### Enviroment Preparation
Please see Wan2.2 (https://github.com/Wan-Video/Wan2.2?tab=readme-ov-file#installation).
### Model Download
| Models              | Download Links                                                                                                                              | Description |
|--------------------|---------------------------------------------------------------------------------------------------------------------------------------------|-------------|
| I2V-A14B    | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B)    🤖 [ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-I2V-A14B)    | Image-to-Video MoE model, supports 480P & 720P |
### Run First-Last-Frame-to-Video Generation
- Single-GPU inference
```sh
python generate.py \
    --task flf2v-A14B \
    --size 832*480 \
    --ckpt_dir ./Wan2.2-I2V-A14B \
    --offload_model False \
    --frame_num 81 \
    --sample_steps 40 \
    --sample_shift 16 \
    --sample_guide_scale 4 \
    --prompt <prompt> \
    --first_frame <first frame path> \
    --last_frame <last frame path> \
    --save_file <output path> \
    --bidirectional_sampling
```
- Multi-GPU inference using FSDP + DeepSpeed Ulysses
```sh
torchrun --nproc_per_node=8 --master_port 39550 generate.py \
    --task flf2v-A14B \
    --size 832*480 \
    --ckpt_dir ./Wan2.2-I2V-A14B \
    --offload_model False \
    --convert_model_dtype \
    --frame_num 81 \
    --sample_steps 40 \
    --sample_shift 16 \
    --sample_guide_scale 4 \
    --dit_fsdp \
    --t5_fsdp \
    --ulysses_size 2 \
    --prompt <prompt> \
    --first_frame <first frame path> \
    --last_frame <last frame path> \
    --save_file <output path> \
    --bidirectional_sampling
```
If you encounter OOM (Out-of-Memory) issues, you can use the `--offload_model True`, `--convert_model_dtype` and `--t5_cpu` options to reduce GPU memory usage.
## Examples

<table>
<tr>
<td>
<video src="https://github.com/user-attachments/assets/5e151a36-3408-4459-b0f5-955a929e1aaf" width=250></video>
</td>
<td>
<video src="https://github.com/user-attachments/assets/fff6b7a8-d8a7-40de-b763-1a72f97fcb28" height=250></video>
</td>
<td>
<video src="https://github.com/user-attachments/assets/3db6cbf9-28e9-4764-bfaf-bd4a7343a117" height=250></video>
</td>
</tr>
</table>

<table>
<tr>
<td>
<video src="https://github.com/user-attachments/assets/be0ee319-8e75-4edc-ac38-1a6194a6a49e" height=250></video>
</td>
<td>
<video src="https://github.com/user-attachments/assets/ccae1fa1-5cab-4eaf-964e-490c9fdd685c" height=250></video>
</td>
<td>
<video src="https://github.com/user-attachments/assets/7a27ca87-dbb8-469e-91a2-b0080bbe6d52" height=250></video>
</td>
</tr>
</table>

<table>
<tr>
<td>
<video src="https://github.com/user-attachments/assets/0a82dbea-4621-4bf1-b65a-5837c3a7ed79" height=250></video>
</td>
<td>
<video src="https://github.com/user-attachments/assets/a5146080-144f-4e19-a6c1-a2ca6e88f5b5" height=250></video>
</td>
<td>
<video src="https://github.com/user-attachments/assets/a3f0acea-504c-4a43-ba21-bcdeddfaa055" height=250></video>
</td>
</tr>
</table>



