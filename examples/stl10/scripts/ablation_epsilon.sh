for seed in 0 1 2
    do
    for eps in 0.4 0.2 0.3 0.6 0.5 0.7
    do
        python main.py --gpus 0 1 --wandb_log --align_eps $eps --run PD_ablation_eps --seed $seed
    done
done