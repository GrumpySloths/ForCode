#!/bin/bash

exp_ids=(11 12 13 14)

for ((i=0; i<${#exp_ids[@]}; i++)); do
    exp_id=${exp_ids[i]}

    session_name="ES_train_${exp_id}"

    tmux kill-session -t $session_name
done
