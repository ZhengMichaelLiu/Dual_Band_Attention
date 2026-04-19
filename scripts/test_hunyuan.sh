SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"/eval/HunyuanVideo
pip install -r requirements.txt
pip install xfuser==0.4.0 --no-deps
pip install av


hf download tencent/HunyuanVideo --local-dir ./ckpts
cd ckpts
hf download xtuner/llava-llama-3-8b-v1_1-transformers --local-dir ./llava-llama-3-8b-v1_1-transformers
cd ..
python hyvideo/utils/preprocess_text_encoder_tokenizer_utils.py --input_dir ckpts/llava-llama-3-8b-v1_1-transformers --output_dir ckpts/text_encoder
cd ckpts
hf download openai/clip-vit-large-patch14 --local-dir ./text_encoder_2
cd ..

# Full attention
python run_vbench.py --num_sampled_prompts 12 --attn flash

# Xattention
python run_vbench.py --num_sampled_prompts 12 --attn xattn --threshold 0.9 --stride 16 --attn_warmup_steps 5
python run_vbench.py --num_sampled_prompts 12 --attn xattn --threshold 0.95 --stride 16 --attn_warmup_steps 5

# Dual_band_attention
python run_vbench.py --num_sampled_prompts 12 --attn dba --threshold 0.9 --stride 16 --attn_warmup_steps 5
python run_vbench.py --num_sampled_prompts 12 --attn dba --threshold 0.95 --stride 16 --attn_warmup_steps 5

# Similarity evaluation
python similarity.py \
  --ref_dirpath  ./results_720p/flash/all_dimension_longer \
  --gen_dirpath  ./results_720p/dba_warmup=5_threshold=0.9_stride=16/all_dimension_longer
  # sampled = 1 : {'psnr': 24.752532958984375, 'lpips': 0.1548604518175125, 'ssim': 0.8094346523284912}
  # sampled = 12: {'psnr': 21.715354919433594, 'lpips': 0.2322687953710556, 'ssim': 0.7602958083152771}

python similarity.py \
  --ref_dirpath  ./results_720p/flash/all_dimension_longer \
  --gen_dirpath  ./results_720p/dba_warmup=5_threshold=0.95_stride=16/all_dimension_longer
  # sampled = 1 : {'psnr': 26.612747192382812, 'lpips': 0.10871648043394089, 'ssim': 0.8664914965629578}
  # sampled = 12: {'psnr': 23.840091705322266, 'lpips': 0.1722966730594635, 'ssim': 0.8139702081680298}

python similarity.py \
  --ref_dirpath  ./results_720p/flash/all_dimension_longer \
  --gen_dirpath  ./results_720p/xattn_warmup=5_threshold=0.9_stride=16/all_dimension_longer
  # sampled = 1 : {'psnr': 24.703556060791016, 'lpips': 0.14414963126182556, 'ssim': 0.8092988729476929}
  # sampled = 12: {'psnr': 21.75971031188965, 'lpips': 0.23283648490905762, 'ssim': 0.759111762046814}

python similarity.py \
  --ref_dirpath  ./results_720p/flash/all_dimension_longer \
  --gen_dirpath  ./results_720p/xattn_warmup=5_threshold=0.95_stride=16/all_dimension_longer
  # sampled = 1 : {'psnr': 26.4310245513916, 'lpips': 0.10863006860017776, 'ssim': 0.8627191185951233}
  # sampled = 12: {'psnr': 23.89733123779297, 'lpips': 0.17002452909946442, 'ssim': 0.8150424957275391}
