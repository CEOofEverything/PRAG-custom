python3 src/augment.py \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --dataset 2wikimultihopqa \
    --data_path data/2wikimultihopqa/ \
    --sample 300  \
    --topk 3

# the training used 2000 data points 
python3 src/warmup_lora.py \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --per_device_train_batch_size 1 \
    --num_train_epochs 1 \
    --learning_rate 3e-4  \
    --lora_rank 2 \
    --lora_alpha 32 \
    --block_size 3000  

python3 src/encode.py \
    --model_name=Qwen/Qwen2.5-7B-Instruct \
    --dataset=2wikimultihopqa \
    --sample=300 \
    --per_device_train_batch_size=1 \
    --num_train_epochs=2 \
    --learning_rate=0.0003 \
    --lora_rank=2 \
    --lora_alpha=32 \
    --with_cot

python3 src/inference.py \
    --model_name=Qwen/Qwen2.5-7B-Instruct \
    --dataset=2wikimultihopqa \
    --sample=300 \
    --num_train_epochs=2 \
    --learning_rate=0.0003 \
    --lora_rank=2 \
    --lora_alpha=32 \
    --max_new_tokens=128 \
    --inference_method=prag \
    --setting 0 1 2 \
    --with_cot
