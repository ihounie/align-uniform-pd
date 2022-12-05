for eps in 0.2 0.4 0.6 0.3 0.5 0.7
do
    python main.py --gpus 0 1 --wandb_log --align_eps $eps --run PD_ablation_eps
done