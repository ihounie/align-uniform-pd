for unif in 0.2 0.8 1.0 1.25
do
for align in 1.0
    do
        python main_penalty.py --gpus 0 1 --wandb_log --run Bline --align_w $align --unif_w $unif
    done
done

for align in 0.2 0.8 1.25
do
    for unif in 1.0
    do
        python main_penalty.py --gpus 0 1 --wandb_log --run Bline --align_w $align --unif_w $unif
    done
done