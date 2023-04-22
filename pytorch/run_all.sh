#!/bin/bash
bash run_fsdp_large.sh train --multi_gpu --eval-interval 500 --log-interval 100 \
    --max_eval_steps 200 --max_step 2000 --batch_size 324 --port 12356 \
    --wrap --chkpt --mem_offload --fp16;

bash run_fsdp_large.sh train --multi_gpu --eval-interval 500 --log-interval 100 \
    --max_eval_steps 200 --max_step 2000 --batch_size 224 --port 12356 \
    --wrap --chkpt;

bash run_fsdp_large.sh train --multi_gpu --eval-interval 500 --log-interval 100 \
    --max_eval_steps 200 --max_step 2000 --batch_size 224 --port 12356 \
    --wrap --chkpt --mem_offload;

bash run_fsdp_large.sh train --multi_gpu --eval-interval 500 --log-interval 100 \
    --max_eval_steps 200 --max_step 2000 --batch_size 180 --port 12356 \
    --chkpt;

bash run_fsdp_large.sh train --multi_gpu --eval-interval 500 --log-interval 100 \
    --max_eval_steps 200 --max_step 2000 --batch_size 180 --port 12356 \
    --wrap --chkpt;