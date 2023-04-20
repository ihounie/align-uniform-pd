for layer in 2 3 4 5
do
    python main_invfirst.py --gpus 0 --layer_inv $layer --wandb_log --knn 12 --run layer_${layer} --lr 0.16238 --project InvarianceFirstSSL --seed 0
done