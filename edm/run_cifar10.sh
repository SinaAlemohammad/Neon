
##### generate synthetic data for fine-tuning, |S| = 6k ############

torchrun --standalone --nproc_per_node=8 generate.py --outdir=datasets/cifar10/ns6k --seeds=0-5999 --batch=64 --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl --steps=18

python make_labels.py --root=datasets/cifar10/ns6k

###### fine tune ###

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  torchrun --standalone --nproc_per_node=8 \
    train.py \
    --outdir=training-runs/cifar10/ns6k \
    --data=datasets/cifar10/ns6k \
    --cond=1 \
    --arch=ddpmpp \
    --lr=1e-4 \
    --transfer=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl \
    --tick=50 \
    --snap=10 \
    --duration=6 \
    --nosubdir

#### evalaute ###

torchrun --standalone --nproc_per_node=8 cal_metrics_all.py \
  --network_pkl_base https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl \
  --network_pkl_aux_dir training-runs/cifar10/ns6k \
  --differential_weights "0.0,0.25,0.5,0.75,1,1.25,1.5,1.75,2" \
  --seeds 50000-99999 \
  --ref https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz \
  --out_dir results/cifar10/ns6k
