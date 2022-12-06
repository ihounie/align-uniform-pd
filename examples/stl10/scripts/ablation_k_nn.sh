for k in 48 24 12 6 3 1
do
    python main_nn.py --gpus 0 1 --wandb_log --knn $k --run PD_ablation_KNN
done