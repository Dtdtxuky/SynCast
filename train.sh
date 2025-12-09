gpus=4
cpus=30

CFG_PATH=''
# configs/base.yaml
# configs/stage1_dpo.yaml
# configs/stage2_spo.yaml

torchrun \
  --nproc_per_node=$gpus \
  --master_port=$PORT \
  train.py \
  --world_size $gpus \
  --cfg "$CFG_PATH" \
  --per_cpus $cpus \
  --tensor_model_parallel_size 1 \
  --outdir '' \
  --resume_cfg_file "$CFG_PATH" \
  --desc ''
