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
from util import seed_everything

def parse_option():
    parser = argparse.ArgumentParser('Representation Learning with Alignment and Uniformity Losses')

    parser.add_argument('--dataset', type=str, default="STL-10", help='dataset') 

    parser.add_argument('--seed', type=int, default=0, help='Seed')
    parser.add_argument('--align_w', type=float, default=0.98, help='Alignment loss initial weight')
    parser.add_argument('--unif_w', type=float, default=0.96, help='Uniformity loss initial weight')
    parser.add_argument('--align_alpha', type=float, default=2, help='alpha in alignment loss')
    parser.add_argument('--unif_t', type=float, default=2, help='t in uniformity loss')
    parser.add_argument('--knn', default=364, type=int, help='Number of Neighbours')

    parser.add_argument('--knn_mode', type=str, default="all", help='either only_k mean_k or all(default)')

    parser.add_argument('--batch_size', type=int, default=768, help='Batch size')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate. Default is linear scaling 0.12 per 256 batch size')
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
    means = {"STL-10": (0.44087801806139126, 0.42790631331699347, 0.3867879370752931),
             "CIFAR-10":(0.49139968, 0.48215827 ,0.44653124)}
    stds = {"STL-10":(0.26826768628079806, 0.2610450402318512, 0.26866836876860795),
             "CIFAR-10": (0.24703233, 0.24348505, 0.26158768)}
    transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(64, scale=(0.08, 1)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
        torchvision.transforms.RandomGrayscale(p=0.2),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            means[opt.dataset],
            stds[opt.dataset],
        ),
    ])
    if opt.dataset =="STL-10":
        torch_dset = torchvision.datasets.STL10(opt.data_folder, 'train+unlabeled', download=True)
    elif opt.dataset=="CIFAR-10":
        torch_dset = torchvision.datasets.CIFAR10(opt.data_folder, 'train', download=True)

    dataset = TwoAugUnsupervisedDataset(torch_dset, transform=transform)
    return torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, num_workers=opt.num_workers,
                                       shuffle=True, pin_memory=True)

def lin_get_data_loaders(opt):
    means = {"STL-10": (0.44087801806139126, 0.42790631331699347, 0.3867879370752931),
             "CIFAR-10":(0.49139968, 0.48215827 ,0.44653124)}
    stds = {"STL-10":(0.26826768628079806, 0.2610450402318512, 0.26866836876860795),
             "CIFAR-10": (0.24703233, 0.24348505, 0.26158768)}
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
            means[opt.dataset],
            stds[opt.dataset],
        ),
    ])
    if opt.dataset =="STL-10":
        torch_dset_train = torchvision.datasets.STL10(opt.data_folder, 'train', download=True)
        torch_dset_test = torchvision.datasets.STL10(opt.data_folder, 'train', download=True)
    elif opt.dataset=="CIFAR-10":
        torch_dset_train = torchvision.datasets.CIFAR10(opt.data_folder, 'test', download=True)
        torch_dset_test = torchvision.datasets.CIFAR10(opt.data_folder, 'test', download=True)
    #train_dataset = torchvision.datasets.STL10(opt.data_folder, 'train', download=True, transform=train_transform)
    #val_dataset = torchvision.datasets.STL10(opt.data_folder, 'test', transform=val_transform)
    train_loader = torch.utils.data.DataLoader(torch_dset_train, batch_size=opt.lin_batch_size,
                                               num_workers=opt.lin_num_workers, shuffle=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(torch_dset_test, batch_size=opt.lin_batch_size,
                                             num_workers=opt.lin_num_workers, pin_memory=True)
    return train_loader, val_loader


def main():
    opt = parse_option()
    seed_everything(opt.seed)
    print(f'Optimize: {opt.align_w:g} * loss_align(alpha={opt.align_alpha:g}) + {opt.unif_w:g} * loss_uniform(t={opt.unif_t:g})')

    torch.cuda.set_device(opt.gpus[0])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    model = SmallAlexNet(feat_dim=opt.feat_dim).to(opt.gpus[0])
    encoder = nn.DataParallel(model, opt.gpus)

    optim = torch.optim.SGD(encoder.parameters(), lr=opt.lr,
                            momentum=opt.momentum, weight_decay=opt.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, gamma=opt.lr_decay_rate,
                                                     milestones=opt.lr_decay_epochs)

    loader = get_data_loader(opt)

    lin_train_loader, lin_val_loader = lin_get_data_loaders(opt)

    align_meter = AverageMeter('align_loss')
    unif_meter = AverageMeter('uniform_loss')
    loss_meter = AverageMeter('total_loss')
    it_time_meter = AverageMeter('iter_time')
    for epoch in range(opt.epochs):
        align_meter.reset()
        unif_meter.reset()
        loss_meter.reset()
        it_time_meter.reset()
        t0 = time.time()
        for ii, (im_x, im_y) in enumerate(loader):
            optim.zero_grad()
            x, y = encoder(torch.cat([im_x.to(opt.gpus[0]), im_y.to(opt.gpus[0])])).chunk(2)
            align_loss_val = align_loss(x, y, alpha=opt.align_alpha)
            unif_loss_val = (uniform_loss(x, t=opt.unif_t, mode = opt.knn_mode, k = opt.knn) + uniform_loss(y, t=opt.unif_t, mode = opt.knn_mode, k = opt.knn)) / 2
            loss = align_loss_val * opt.align_w + unif_loss_val * opt.unif_w
            align_meter.update(align_loss_val, x.shape[0])
            unif_meter.update(unif_loss_val)
            loss_meter.update(loss, x.shape[0])
            loss.backward()
            optim.step()
            it_time_meter.update(time.time() - t0)
            if ii % opt.log_interval == 0:
                print(f"Epoch {epoch}/{opt.epochs}\tIt {ii}/{len(loader)}\t" +
                      f"{align_meter}\t{unif_meter}\t{loss_meter}\t{it_time_meter}")
                if opt.wandb_log:
                    wandb.log({"epoch":epoch, "align_loss": align_meter.avg, "uniform_loss":unif_meter.avg})
            t0 = time.time()
        scheduler.step()
        if epoch % opt.lin_eval_interval == 0 and epoch>0:
            t_val = time.time()
            model_eval = deepcopy(model).eval()
            val_acc = train_linear(model_eval, lin_train_loader, lin_val_loader, opt)
            print(f"val acc {val_acc}, time: {time.time()-t_val}")
            wandb.log({"epoch":epoch, "val acc":val_acc})
            encoder.train()
    ckpt_file = os.path.join(opt.save_folder, 'encoder.pth')
    torch.save(encoder.module.state_dict(), ckpt_file)
    print(f'Saved to {ckpt_file}')
    if opt.wandb_log:
        wandb.save(ckpt_file, policy = 'now')
    model.eval()
    val_acc = train_linear(model, lin_train_loader, lin_val_loader, opt)
    print(f"final val acc {val_acc}")
    if opt.wandb_log:
        wandb.log({"final val acc":val_acc})

if __name__ == '__main__':
    main()
