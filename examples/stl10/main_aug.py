import os
import time
import argparse

import torchvision
import torch
import torch.nn as nn
import wandb
from copy import deepcopy

from util import AverageMeter, TwoAugUnsupervisedDataset
from encoder import SmallAlexNet
from align_uniform import align_loss, uniform_loss
from linear_eval import train_linear
torch.cuda.empty_cache()

def parse_option():
    parser = argparse.ArgumentParser('STL-10 Representation Learning with Alignment and Uniformity Losses')

    parser.add_argument('--align_w', type=float, default=1.0, help='Alignment loss initial weight')
    parser.add_argument('--unif_w', type=float, default=1.0, help='Uniformity loss initial weight')
    parser.add_argument('--align_eps', type=float, default=0.4, help='Alignment Loss Epsilon')
    parser.add_argument('--align_alpha', type=float, default=2, help='alpha in alignment loss')
    parser.add_argument('--unif_t', type=float, default=2, help='t in uniformity loss')

    parser.add_argument('--batch_size', type=int, default=500, help='Batch size')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate. Default is linear scaling 0.12 per 256 batch size')
    parser.add_argument('--lr_dual', type=float, default=0.2, help='Dual Learning rate')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='Learning rate decay rate')
    parser.add_argument('--lr_decay_epochs', default=[155, 170, 185], nargs='*', type=int,
                        help='When to decay learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='L2 weight decay')
    parser.add_argument('--feat_dim', type=int, default=128, help='Feature dimensionality')

    parser.add_argument('--num_workers', type=int, default=20, help='Number of data loader workers to use')
    parser.add_argument('--log_interval', type=int, default=40, help='Number of iterations between logs')
    parser.add_argument('--gpus', default=[0], nargs='*', type=int,
                        help='List of GPU indices to use, e.g., --gpus 0 1 2 3')

    parser.add_argument('--data_folder', type=str, default='./data', help='Path to data')
    parser.add_argument('--result_folder', type=str, default='./results', help='Base directory to save model')

    parser.add_argument('--wandb_log',  action='store_true')
    parser.add_argument('--project',  default='Uniformity', type=str, help='wandb Project name')
    parser.add_argument('--run',  default='PD', type=str, help='wandb Run name')
    
    parser.add_argument('--lin_layer_index', type=int, default=-2, help='Evaluation layer')

    parser.add_argument('--lin_batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lin_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lin_lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--lin_lr_decay_rate', type=float, default=0.2, help='Learning rate decay rate')
    parser.add_argument('--lin_lr_decay_epochs', type=str, default='60,80', help='When to decay learning rate')

    parser.add_argument('--lin_num_workers', type=int, default=6, help='Number of data loader workers to use')
    parser.add_argument('--lin_log_interval', type=int, default=40, help='Number of iterations between logs')
    parser.add_argument('--lin_eval_interval', type=int, default=400, help='Number of epochs between linear evaluations')
    opt = parser.parse_args()

    if opt.lr is None:
        opt.lr = 0.12 * (opt.batch_size / 256)

    opt.gpus = list(map(lambda x: torch.device('cuda', x), opt.gpus))

    opt.save_folder = os.path.join(
        opt.result_folder,
        f"align{opt.align_w:g}alpha{opt.align_alpha:g}_unif{opt.unif_w:g}t{opt.unif_t:g}"
    )
    os.makedirs(opt.save_folder, exist_ok=True)
    if opt.wandb_log:
        wandb.init(project=opt.project, entity="hounie", name=opt.run)
        wandb.config.update(opt)

    return opt


def get_data_loader(opt):
    all_loaders = {}
    for aug, name in zip([torchvision.transforms.RandomHorizontalFlip(p=0.5),
                        torchvision.transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
                        torchvision.transforms.RandomGrayscale(p=0.5)],
                        ["flip", "jitter", "gray"]):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(64, scale=(0.08, 1)),
            aug,
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.44087801806139126, 0.42790631331699347, 0.3867879370752931),
                (0.26826768628079806, 0.2610450402318512, 0.26866836876860795),
            ),
        ])
        dataset = TwoAugUnsupervisedDataset(
            torchvision.datasets.STL10(opt.data_folder, 'train+unlabeled', download=True), transform=transform)
        all_loaders[name] =  torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, num_workers=opt.num_workers,
                                       shuffle=True, pin_memory=True)
    return all_loaders

def lin_get_data_loaders(opt):
    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(64, scale=(0.08, 1)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            (0.44087801806139126, 0.42790631331699347, 0.3867879370752931),
            (0.26826768628079806, 0.2610450402318512, 0.26866836876860795),
        ),
    ])
    val_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(70),
        torchvision.transforms.CenterCrop(64),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            (0.44087801806139126, 0.42790631331699347, 0.3867879370752931),
            (0.26826768628079806, 0.2610450402318512, 0.26866836876860795),
        ),
    ])
    train_dataset = torchvision.datasets.STL10(opt.data_folder, 'train', download=True, transform=train_transform)
    val_dataset = torchvision.datasets.STL10(opt.data_folder, 'test', transform=val_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.lin_batch_size,
                                               num_workers=opt.lin_num_workers, shuffle=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=opt.lin_batch_size,
                                             num_workers=opt.lin_num_workers, pin_memory=True)
    return train_loader, val_loader


def main():
    opt = parse_option()

    print(f'Optimize: {opt.align_w:g} * loss_align(alpha={opt.align_alpha:g}) + {opt.unif_w:g} * loss_uniform(t={opt.unif_t:g})')

    torch.cuda.set_device(opt.gpus[0])
    #torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    model = SmallAlexNet(feat_dim=opt.feat_dim).to(opt.gpus[0])
    encoder = nn.DataParallel(model, opt.gpus)

    optim = torch.optim.SGD(encoder.parameters(), lr=opt.lr,
                            momentum=opt.momentum, weight_decay=opt.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, gamma=opt.lr_decay_rate,
                                                     milestones=opt.lr_decay_epochs)

    loader = get_data_loader(opt)
    loader_flip, loader_jitter, loader_gray = deepcopy(loader["flip"]), deepcopy(loader["jitter"]), deepcopy(loader["gray"])

    lin_train_loader, lin_val_loader = lin_get_data_loaders(opt)

    align_meter = [AverageMeter('flip'), AverageMeter('jitter'), AverageMeter('gray')]
    unif_meter = AverageMeter('uniform_loss')
    loss_meter = AverageMeter('total_loss')
    it_time_meter = AverageMeter('iter_time')

    dual_var = [1.0, 1.0, 1.0] # I avoid using a tensor to circumvent parallelization issues
    for epoch in range(opt.epochs):
        for m in align_meter:
            m.reset()
        unif_meter.reset()
        loss_meter.reset()
        it_time_meter.reset()
        t0 = time.time()
        for ii, ((im_x_flip, im_y_flip), (im_x_jitter, im_y_jitter), (im_x_gray, im_y_gray)) in enumerate(zip(loader_flip, loader_jitter, loader_gray)):
            optim.zero_grad()
            b_size = im_x_flip.shape[0]
            x_flip, y_flip, x_jitter, y_jitter, x_gray, y_gray = encoder(
                                        torch.cat([im_x_flip.to(opt.gpus[0]), im_y_flip.to(opt.gpus[0]),
                                        im_x_jitter.to(opt.gpus[0]), im_y_jitter.to(opt.gpus[0]),
                                        im_x_gray.to(opt.gpus[0]), im_y_gray.to(opt.gpus[0])])).chunk(6)
            unif_loss_val = (uniform_loss(x_flip, t=opt.unif_t) + uniform_loss(y_flip, t=opt.unif_t)
                            + uniform_loss(x_jitter, t=opt.unif_t) + uniform_loss(y_jitter, t=opt.unif_t)
                            + uniform_loss(x_gray, t=opt.unif_t) + uniform_loss(y_gray, t=opt.unif_t)
                            ) / 6
            align_flip, align_jitter, align_gray =  align_loss(x_flip, y_flip, alpha=opt.align_alpha),\
                                                    align_loss(x_jitter, y_jitter, alpha=opt.align_alpha),\
                                                    align_loss(x_gray, y_gray, alpha=opt.align_alpha)
            loss = align_flip * dual_var[0] \
                 + align_jitter * dual_var[1] \
                 + align_gray * dual_var[2] \
                 + unif_loss_val
            align_meter[0].update(align_flip.item(), b_size)
            align_meter[1].update(align_jitter.item(), b_size)
            align_meter[2].update(align_gray.item(), b_size)
            unif_meter.update(unif_loss_val.item())
            loss_meter.update(loss.item(), b_size)
            loss.backward()
            optim.step()
            it_time_meter.update(time.time() - t0)
            if ii % opt.log_interval == 0:
                print(f"Epoch {epoch}/{opt.epochs}\tIt {ii}/{len(loader)}\t" +
                      f"a: {align_meter[0]} {align_meter[1]} {align_meter[2]}\t u: {unif_meter}\t l: {loss_meter}\t{it_time_meter}")
                if opt.wandb_log:
                    wandb.log({"epoch":epoch, "align_loss_flip": align_meter[0].avg, 
                    "align_loss_jitter": align_meter[1].avg, "align_loss_gray": align_meter[1].avg, "uniform_loss":unif_meter.avg})
            t0 = time.time()
        scheduler.step()
        with torch.no_grad():
            for m in align_meter:
                m.reset()
            encoder.eval()
            for ii, ((im_x_flip, im_y_flip), (im_x_jitter, im_y_jitter), (im_x_gray, im_y_gray)) in enumerate(zip(loader["flip"], loader["jitter"], loader["gray"])):
                b_size = im_x_flip.shape[0]
                x_flip, y_flip, x_jitter, y_jitter, x_gray, y_gray = encoder(
                                            torch.cat([im_x_flip.to(opt.gpus[0]), im_y_flip.to(opt.gpus[0]),
                                            im_x_jitter.to(opt.gpus[0]), im_y_jitter.to(opt.gpus[0]),
                                            im_x_gray.to(opt.gpus[0]), im_y_gray.to(opt.gpus[0])])).chunk(6)
                align_meter[0].update(align_loss(x_flip, y_flip, alpha=opt.align_alpha), b_size)
                align_meter[1].update(align_loss(x_jitter, y_jitter, alpha=opt.align_alpha), b_size)
                align_meter[2].update(align_loss(x_gray, y_gray, alpha=opt.align_alpha), b_size)
            
            slack = [m.avg - opt.align_eps for m in align_meter]
            dual_var = [max(0,d + s) for d, s in zip(dual_var, slack)]
            print(f"dual var {dual_var}, slack {slack}")
            if opt.wandb_log:
                wandb.log({"epoch":epoch, 
                            "dual_var_flip": dual_var[0],"dual_var_jitter": dual_var[1],"dual_var_gray": dual_var[2],
                            "slack_flip":slack[0], "slack_jitter":slack[1],"slack_gray": slack[2]})
        if epoch % opt.lin_eval_interval == 0 and epoch>0:
            t_val = time.time()
            model_eval = deepcopy(model).eval()
            val_acc = train_linear(model_eval, lin_train_loader, lin_val_loader, opt)
            print(f"Val Acc {val_acc}, time: {time.time()-t_val}")
            wandb.log({"epoch":epoch, "val acc":val_acc})
        encoder.train()
    ckpt_file = os.path.join(opt.save_folder, 'encoder.pth')
    torch.save(encoder.module.state_dict(), ckpt_file)
    print(f'Saved to {ckpt_file}')
    model.eval()
    val_acc = train_linear(model, lin_train_loader, lin_val_loader, opt)
    print(f"final val acc {val_acc}")
    wandb.log({"final val acc":val_acc})


if __name__ == '__main__':
    main()
