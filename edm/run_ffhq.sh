

##### generate synthetic data for fine-tuning, |S| = 6k ############

torchrun --standalone --nproc_per_node=8 generate.py --outdir=datasets/ffhq/ns18k --seeds=0-17999 --batch=64 --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-ffhq-64x64-uncond-vp.pkl --steps=40


###### fine tune ###

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  torchrun --standalone --nproc_per_node=8 \
    train.py \
    --outdir=training-runs/ffhq/ns18k \
    --data=datasets/ffhq/ns18k \
    --cond=0 \
    --arch=ddpmpp \
    --batch=256 \
    --cres=1,2,2,2 \
    --lr=4e-6 \
    --dropout=0.05 \
    --augment=0.15 \
    --transfer=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-ffhq-64x64-uncond-vp.pkl \
    --tick=50 \
    --snap=5 \
    --duration=3 \
    --nosubdir

#### evalaute ###

torchrun --standalone --nproc_per_node=8 cal_metrics_all.py \
  --network_pkl_base https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-ffhq-64x64-uncond-vp.pkl \
  --network_pkl_aux_dir training-runs/ffhq/ns18k \
  --differential_weights "0.0,0.25,0.5,0.75,1,1.25,1.5,1.75,2,2.25,2.5,2.75,3" \
  --seeds 500000-549999 \
  --num_steps=40 \
  --ref https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/ffhq-64x64.npz \
  --out_dir results/ffhq/ns18k

