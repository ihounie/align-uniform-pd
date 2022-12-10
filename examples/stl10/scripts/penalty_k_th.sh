for seed in 0 1 2
do
    for k in 12 1 24 48 96 192 6 3
    do
        python main_penalty.py --knn_mode only_k --gpus 0 1 --wandb_log --knn $k --run Bline_Kth --seed $seed
    done
done