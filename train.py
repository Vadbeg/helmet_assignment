from __future__ import absolute_import, division, print_function

import json
import os

import torch
import torch.utils.data
from torchvision.transforms import transforms as T

from helmet_assignment.modules.datasets.jde import JointDatasetHelmet
from helmet_assignment.modules.logger import Logger
from helmet_assignment.modules.models.model import create_model, load_model, save_model
from helmet_assignment.modules.opts import opts
from helmet_assignment.modules.train.train_factory import train_factory


def main(opt):
    torch.manual_seed(opt.seed)
    torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test

    print('Setting up data...')
    transforms = T.Compose([T.ToTensor()])

    dataset_root = (
        '/home/vadbeg/Data_SSD/Kaggle/helmet/' 'nfl-health-and-safety-helmet-assignment'
    )

    dataset = JointDatasetHelmet(
        opt, dataset_root, (864, 480), augment=True, transforms=transforms
    )
    opt = opts().update_dataset_info_and_set_heads(opt, dataset)
    print(opt)

    logger = Logger(opt)

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')

    print('Creating model...')
    model = create_model(opt.arch, opt.heads, opt.head_conv)
    optimizer = torch.optim.Adam(model.parameters(), opt.lr)
    start_epoch = 0

    # Get dataloader

    print('Batch size:')
    print(opt.batch_size)

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    print('Starting training...')
    Trainer = train_factory[opt.task]
    trainer = Trainer(opt, model, optimizer)
    trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)

    if opt.load_model != '':
        model, optimizer, start_epoch = load_model(
            model, opt.load_model, trainer.optimizer, opt.resume, opt.lr, opt.lr_step
        )

    for epoch in range(start_epoch + 1, opt.num_epochs + 1):
        mark = epoch if opt.save_all else 'last'
        log_dict_train, _ = trainer.train(epoch, train_loader)
        logger.write('epoch: {} |'.format(epoch))
        for k, v in log_dict_train.items():
            logger.scalar_summary('train_{}'.format(k), v, epoch)
            logger.write('{} {:8f} | '.format(k, v))

        if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
            save_model(
                os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)),
                epoch,
                model,
                optimizer,
            )
        else:
            save_model(
                os.path.join(opt.save_dir, 'model_last.pth'), epoch, model, optimizer
            )
        logger.write('\n')
        if epoch in opt.lr_step:
            save_model(
                os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
                epoch,
                model,
                optimizer,
            )
            lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
            print('Drop LR to', lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        if epoch % 5 == 0 or epoch >= 25:
            save_model(
                os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
                epoch,
                model,
                optimizer,
            )
    logger.close()


if __name__ == '__main__':
    torch.cuda.set_device(0)
    opt = opts().parse()
    main(opt)
