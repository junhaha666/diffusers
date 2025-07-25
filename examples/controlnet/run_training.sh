	export HIP_VISIBLE_DEVICES=6
	export MODEL_NAME=/data/jun/stable-diffusion-3.5-medium 
	export OUTPUT_DIR=/data/jun/stable-diffusion-3.5-medium-finetune_jun
	export DATASET_NAME=fusing/fill50k
    export DATASET_DIR=/mnt/raid0/jun/mydocker/model/dataset/fusing_fill50k_process/train
	
	accelerate launch --mixed_precision="bf16"  train_controlnet_sd3.py \
	  --pretrained_model_name_or_path=$MODEL_NAME \
	  --dataset_name=$DATASET_NAME \
	  --output_dir=$OUTPUT_DIR \
	  --resolution=1024 \
	  --learning_rate=1e-5 \
	  --max_train_steps=150 \
	  --validation_image "./conditioning_image_1.png" "./conditioning_image_2.png" \
	  --validation_prompt "red circle with blue background" "cyan circle with brown floral background" \
	  --validation_steps=100 \
	  --train_batch_size=4 \
	  --dataloader_num_workers 8 \
	  --dataset_preprocess_batch_size 1000 \
	  --gradient_accumulation_steps=4
