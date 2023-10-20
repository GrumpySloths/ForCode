exp_id=7
body_gain=1.0
foot_gain=1.0
model_id=670209
cd ..
python ES_train.py --exp_id $exp_id --body_gain $body_gain \
 --foot_gain $foot_gain --eval 1 --eval_ModelID $model_id --debug 1