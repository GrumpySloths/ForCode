#!/bin/bash

exp_ids=(11 12 13 14)
cuda_devices=(4 5 6 7)
# foot_gains=(0.4 0.6 0.8 1.0)
body_gains=(0.2 0.4 0.6 0.8)


for ((i=0; i<${#exp_ids[@]}; i++)); do
    exp_id=${exp_ids[i]}
    cuda_device=${cuda_devices[i]}
    # foot_gain=${foot_gains[i]}
    body_gain=${body_gains[i]}
    session_name="ES_train_${exp_id}"

    tmux kill-session -t $session_name
    tmux new -s $session_name -d
    echo "创建$session_name 成功"
    tmux send-keys -t $session_name "conda activate pytorch" Enter
    tmux send-keys -t $session_name "export CUDA_VISIBLE_DEVICES=$cuda_device" Enter
    tmux send-keys -t $session_name "cd /data/niujh/ForCode/NewMujo_test/src" Enter
    # tmux send-keys -t $session_name "python ES_train.py --exp_id $exp_id --foot_gain $foot_gain" Enter
    tmux send-keys -t $session_name "python ES_train.py --exp_id $exp_id --body_gain $body_gain --eval 0" Enter
done
