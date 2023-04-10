for seed in 0 1 2
do
    for l in 3 5 7 4 6 8
    do
        python main_nas.py --lr 0.16238 --gpus 0 1 --wandb_log --knn 12 --run k12_l_${l} --seed $seed --num_layers $l
    done
done