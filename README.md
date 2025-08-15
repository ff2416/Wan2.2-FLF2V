# WanFM
<p align="center">
<img height="250" alt="logo" src="https://github.com/user-attachments/assets/5c38e92c-727f-47e4-84b6-5df3f0121aa8" />
<p>
<p align="center">
Pengjun Fang, Harry Yang
</p>
In this repository, we present WanFM, which builds upon Wan 2.2 Image-to-Video (I2V) and introduces several key enhancements:

- **Last Frame Constraint**: Enforces precise alignment between the generated last frame and the target frame, ensuring consistent video endpoints.

- **Bidirectional Denoising with Time Reversal Fusion**: Performs denoising both forward (first-to-last) and backward (last-to-first), fusing intermediate results at every step for superior temporal coherence. To accommodate bidirectional fusion, the original denoising formula—where each step depends on the previous one—has been redesigned, allowing non-continuous, step-wise integration of forward and backward denoised states.

- **Prompt-Adapted Temporal Attention**: During the reverse pass, temporal self-attention is rotated to align backward generation with the prompt, enabling bidirectionally refined, prompt-consistent video sequences.

With these improvements, we achieve First–Last–Frame-to-Video generation (FLF2V), enabling controllable and consistent video synthesis given the first and last frames as constraints.
## Demo
<div align="center">
  <video src="https://github.com/user-attachments/assets/0e72ac8b-75a2-47fa-9530-c89650621e92" width="100%" poster=""> </video>
</div>

## Run WanFM
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
## More Examples

<table>
<tr>
    <td> 
        <img height="300" alt="flf2v_input_first_frame" src="https://github.com/user-attachments/assets/04ffb205-7f8a-44af-9185-3ba8f0c12da3" /> 
        <img height="300" alt="flf2v_input_last_frame" src="https://github.com/user-attachments/assets/270d5dec-2483-41b8-b6bd-0961ed002712" /> 
    </td>
    <td> 
        <video src="https://github.com/user-attachments/assets/5e151a36-3408-4459-b0f5-955a929e1aaf" width="100%" poster=""></video> 
    </td>
    <td> 
        <img height="300" alt="lmotion1_0" src="https://github.com/user-attachments/assets/bff1313c-d3ac-465d-830d-bef1932e95c8" />
        <img height="300" alt="lmotion1_1" src="https://github.com/user-attachments/assets/b8efb31a-ba0d-4128-9ca4-9392c4fc1ce8" />
    </td>   
    <td> 
        <video src="https://github.com/user-attachments/assets/fff6b7a8-d8a7-40de-b763-1a72f97fcb28" width="100%" poster=""></video> 
    </td>
</tr>
</table>

<table>
<tr>
    <td>
        <img width="300" height="597" alt="002" src="https://github.com/user-attachments/assets/f9a9e103-6764-42f4-8715-4ebfe7618f8b" />
        <img width="300" height="656" alt="003" src="https://github.com/user-attachments/assets/6f315086-aea7-486f-afaa-0fa426825761" />
    </td>
    <td> 
        <video src="https://github.com/user-attachments/assets/466fcb51-e765-45da-8920-627066a432fc" width="100%"></video> 
    </td>
    <td>
        <img height="300" alt="pusa0_0" src="https://github.com/user-attachments/assets/aa2ec8c8-2e48-490b-8575-bf2861094396" />
        <img height="300" alt="pusa0_1" src="https://github.com/user-attachments/assets/a7c2a2d0-371d-4945-9688-e61ddf7cc583" />
    </td>
    <td>
        <video src="https://github.com/user-attachments/assets/be0ee319-8e75-4edc-ac38-1a6194a6a49e" height=250></video>
    </td>
</tr>
</table>

<table>
<tr>
    <td>
        <img height="300" alt="pusa1_0" src="https://github.com/user-attachments/assets/a06b98fa-1872-4417-9a9b-6378d6e172ef" />
        <img height="300" alt="pusa1_1" src="https://github.com/user-attachments/assets/05d70729-5653-4299-93fd-b25dca7bf948" />
    </td>
    <td>
        <video src="https://github.com/user-attachments/assets/ccae1fa1-5cab-4eaf-964e-490c9fdd685c" height=250></video>
    </td>
    <td>
        <img height="300" alt="pusa2_0" src="https://github.com/user-attachments/assets/0693a15d-0e23-4fcd-810d-5639962b096f" />
        <img height="300" alt="pusa2_1" src="https://github.com/user-attachments/assets/49da0ef2-f50e-415f-9571-df40c991b0ec" />
    </td>
    <td>
        <video src="https://github.com/user-attachments/assets/7a27ca87-dbb8-469e-91a2-b0080bbe6d52" height=250></video>
    </td>
</tr>
</table>

<table>
<tr>
    <td>
        <img height="300" alt="pusa3_0" src="https://github.com/user-attachments/assets/6aaf88f2-d19b-4605-b9d3-1a7938445fac" />
        <img height="300" alt="pusa3_1" src="https://github.com/user-attachments/assets/2f723020-16ff-43cb-9aae-355a11975b01" />
    </td>
    <td>
        <video src="https://github.com/user-attachments/assets/0a82dbea-4621-4bf1-b65a-5837c3a7ed79" height=250></video>
    </td>
    <td>
        <img height="300" alt="cola1" src="https://github.com/user-attachments/assets/5e184b40-6d1f-443a-a858-f3e648ed7399" />
        <img height="300" alt="cola2" src="https://github.com/user-attachments/assets/62c8bf0a-685b-4ad1-9512-220e3c6f0845" />
    </td>
    <td>
        <video src="https://github.com/user-attachments/assets/a5146080-144f-4e19-a6c1-a2ca6e88f5b5" height=250></video>
    </td>
</tr>
</table>

<table>
<tr>
    <td>
        <img height="300" alt="pusa4_0" src="https://github.com/user-attachments/assets/58f2cd9a-61df-41f0-b69b-6c281b46fa3a" />
        <img height="300" alt="pusa4_1" src="https://github.com/user-attachments/assets/2dd3f40d-dcad-4148-b45c-6c48c101fdac" />
    </td>
    <td>
        <video src="https://github.com/user-attachments/assets/a3f0acea-504c-4a43-ba21-bcdeddfaa055" height=250></video>
    </td>
    <td>
        <img height="300" alt="huang0" src="https://github.com/user-attachments/assets/9f0d6f2e-f6a1-47af-9858-24ab8d26536a" />
        <img height="300" alt="huang1" src="https://github.com/user-attachments/assets/73d3b5c3-a513-4650-ab45-e6b3bf670c13" />
    </td>
    <td>
        <video src="https://github.com/user-attachments/assets/9ed3b4d7-7350-471e-8011-3b5e2d247783" height=250></video>
    </td>
</tr>
</table>



