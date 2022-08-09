# Decision Transformer (DT) reward_conditioned
for horizon in 5
do
    python run_dt_atari.py --seed 123 --horizon $horizon --context_length 5 --epochs 300 --model_type 'naive' --num_steps 50000 --num_buffers 50 --game 'Breakout' --batch_size 64
done