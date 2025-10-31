#!/bin/bash

torchrun --nproc-per-node=4 --master-addr="localhost" --master-port=12355 test_gpu_freqs.py --gpu_freqs 1530 1500 1402 1297 1200 1102 997 900 877 802 697 600 502 397 300 --task all_reduce --all_reduce_num_comm_per_iteration 225
torchrun --nproc-per-node=4 --master-addr="localhost" --master-port=12355 test_gpu_freqs.py --gpu_freqs 1530 1500 1402 1297 1200 1102 997 900 877 802 697 600 502 397 300 --task a2a_single --a2a_single_num_comm_per_iteration 225
torchrun --nproc-per-node=2 --master-addr="localhost" --master-port=12355 test_gpu_freqs.py --gpu_freqs 1530 1500 1402 1297 1200 1102 997 900 877 802 697 600 502 397 300 --task sr --sr_num_comm_per_iteration 100
python3 test_gpu_freqs.py --gpu_freqs 1530 1500 1402 1297 1200 1102 997 900 877 802 697 600 502 397 300 --task mm --mm_num_op_per_iteration 500 --mm_in_dimension 2048 --mm_intermediate_dimension 768 --mm_out_dimension 2048
python3 test_gpu_freqs.py --gpu_freqs 1530 1500 1402 1297 1200 1102 997 900 877 802 697 600 502 397 300 --task tadd --tadd_num_op_per_iteration 1000 --tadd_num_data_reuse 5 --tadd_matrix 2048 2048
python3 test_gpu_freqs.py --gpu_freqs 1530 1500 1402 1297 1200 1102 997 900 877 802 697 600 502 397 300 --task tcopy --tcopy_num_op_per_iteration 250 --tcopy_num_data_reuse 5
python3 test_gpu_freqs.py --gpu_freqs 1530 1500 1402 1297 1200 1102 997 900 877 802 697 600 502 397 300 --task tsa --tsa_num_data_reuse 3 --tsa_num_op_per_iteration 500
python3 test_gpu_freqs.py --gpu_freqs 1530 1500 1402 1297 1200 1102 997 900 877 802 697 600 502 397 300 --task tss --tss_num_data_reuse 3 --tss_num_op_per_iteration 500
python3 test_gpu_freqs.py --gpu_freqs 1530 1500 1402 1297 1200 1102 997 900 877 802 697 600 502 397 300 --task tsm --tsm_num_data_reuse 3 --tsm_num_op_per_iteration 500
python3 test_gpu_freqs.py --gpu_freqs 1530 1500 1402 1297 1200 1102 997 900 877 802 697 600 502 397 300 --task tsd --tsd_num_data_reuse 3 --tsd_num_op_per_iteration 500
python3 test_gpu_freqs.py --gpu_freqs 1530 1500 1402 1297 1200 1102 997 900 877 802 697 600 502 397 300 --task act_fn --act_fn_function relu --act_fn_num_data_reuse 6 --act_fn_num_op_per_iteration 60 --act_fn_numel 67108864
python3 test_gpu_freqs.py --gpu_freqs 1530 1500 1402 1297 1200 1102 997 900 877 802 697 600 502 397 300 --task act_fn --act_fn_function gelu --act_fn_num_data_reuse 6 --act_fn_num_op_per_iteration 60 --act_fn_numel 67108864
python3 test_gpu_freqs.py --gpu_freqs 1530 1500 1402 1297 1200 1102 997 900 877 802 697 600 502 397 300 --task act_fn --act_fn_function silu --act_fn_num_data_reuse 6 --act_fn_num_op_per_iteration 60 --act_fn_numel 67108864
python3 test_gpu_freqs.py --gpu_freqs 1530 1500 1402 1297 1200 1102 997 900 877 802 697 600 502 397 300 --task h2d --h2d_num_ops_per_iteration 10 --h2d_numel 67108864
python3 test_gpu_freqs.py --gpu_freqs 1530 1500 1402 1297 1200 1102 997 900 877 802 697 600 502 397 300 --task d2h --d2h_num_ops_per_iteration 3 --d2h_numel 268435456
