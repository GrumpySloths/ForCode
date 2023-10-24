exp_id=18
body_gain=0.2
foot_gain=1
model_id=840079
cd ..
python ES_train.py --exp_id $exp_id --body_gain $body_gain \
 --foot_gain $foot_gain --eval 1 --eval_ModelID $model_id --debug 1