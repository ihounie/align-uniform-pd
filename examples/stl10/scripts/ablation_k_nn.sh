for k in 12 1 24 6 3 48 384 192 96
do
    for seed in 0 1 2 3
    do
        python main_nn.py --gpus 0 1 --wandb_log --knn $k --run PD_ablation_KNN
    done
done