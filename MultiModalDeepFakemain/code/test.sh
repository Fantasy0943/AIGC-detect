EXPID=your_best_model_dir_name

HOST='127.0.0.1'
PORT='1'

NUM_GPU=1

python test.py \
--config 'configs/test.yaml' \
--output_dir 'results' \
--launcher pytorch \
--rank 0 \
--log_num 20240709 \
--token_momentum \
--world_size 0 \
--test_epoch best \

