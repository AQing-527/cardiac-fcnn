import os
import time
import argparse
import json
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataset import EchoData
from models.fcnn import FCNN
import utils


class Trainer(object):
    def __init__(self, config, ckpt_path=None):
        # continue training
        self.resume = False
        if ckpt_path is not None and ckpt_path != '':
            self.resume = True
            try:
                self.ckpt = torch.load(ckpt_path)
            except Exception:
                print('Error: cannot load checkpoint!')
                exit()

        self.init_time = utils.current_time()
        if self.resume:
            self.init_time = self.ckpt['init_time']

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.struct = config['struct']
        self.print_interval = config['print_interval']
        self.save_interval = config['save_interval']
        self.log_interval = config['log_interval']

        self.pth_path = os.path.join(f'pths/{self.struct}', 'transfer')
        self.log_path = os.path.join(f'logs/{self.struct}', 'transfer')
        os.makedirs(self.pth_path, exist_ok=True)
        os.makedirs(self.log_path, exist_ok=True)
        self.logger = SummaryWriter(self.log_path)
        self.console_path = os.path.join(self.log_path, 'console_history.txt')
        if not self.resume:
            with open(os.path.join(self.pth_path, '_CONFIG.json'), 'w') as f:
                json.dump(config, f, indent=4)
            with open(os.path.join(self.log_path, '_CONFIG.json'), 'w') as f:
                json.dump(config, f, indent=4)
            console_file = open(self.console_path, 'w')
            console_file.close()

        self.train_data = EchoData(f'data/meta/train/{self.struct}', norm_echo=True, augmentation=False)
        self.val_data = EchoData(f'data/meta/val/{self.struct}', norm_echo=True, augmentation=False)

        self.train_loader = DataLoader(
            self.train_data, batch_size=config['batch_size'], shuffle=True, drop_last=False, num_workers=8)
        self.val_loader = DataLoader(
            self.val_data, batch_size=config['batch_size'], shuffle=False, drop_last=False, num_workers=8)

        self.epochs = config['epochs']
        self.model = FCNN().to(self.device)
        if self.resume:
            self.model.load_state_dict(self.ckpt['model_state_dict'])
        self.loss_fn1 = nn.L1Loss().to(self.device)
        self.loss_fn2 = nn.BCELoss().to(self.device)
        
        self.optimizer = optim.AdamW(self.model.parameters(
        ), lr=config['lr'], weight_decay=config['weight_decay'], amsgrad=True)
        # if self.resume:
        #     self.optimizer.load_state_dict(self.ckpt['optimizer_state_dict'])
        self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=config['factor'], patience=config['patience'])
        # if self.resume:
        #     self.lr_scheduler.load_state_dict(
        #         self.ckpt['lr_scheduler_state_dict'])
        self.total_train_step = 0
        self.total_val_step = 0
        self.start_time = 0.0
        self.end_time = 0.0
        self.last_val_loss = float('inf')
        self.best_epoch = 0
        if self.resume:
            self.total_train_step = self.ckpt['total_train_step']
            self.total_val_step = self.ckpt['total_val_step']
            # self.last_val_loss = self.ckpt['last_val_loss']
            self.best_epoch = self.ckpt['best_epoch']

    def train(self):
        self.print('Train loss:')
        self.model.train()
        size = len(self.train_loader.dataset)
        train_loss = 0

        for batch, (filename, echo, displacement_vector, classifier) in enumerate(self.train_loader):
            classifier = classifier.reshape(-1,1)
            echo, displacement_vector, classifier = echo.to(self.device), displacement_vector.to(self.device), classifier.to(self.device)
            pred = self.model(echo)
            pred_displacement = pred[0]
            pred_classifier = pred[1]
            loss = self.loss_fn1(pred_displacement, displacement_vector) + self.loss_fn2(pred_classifier, classifier)
            loss /= len(echo)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            train_loss += loss.item() * len(echo)
            if batch % self.print_interval == 0:
                loss_value, curr = loss.item(), batch*len(echo)
                self.print(f'train: {loss_value:.9e} [{curr:>3d}/{size:>3d}]')

            self.total_train_step += 1
            if self.total_train_step % self.log_interval == 0:
                self.logger.add_scalar(
                    'training loss', loss.item(), self.total_train_step)

        train_loss /= size
        self.print(f'train: {train_loss:.9e} [average]')
        return train_loss

    def eval(self):
        self.print('Val loss:')
        self.model.eval()
        size = len(self.val_loader.dataset)
        val_loss = 0.0

        with torch.no_grad():
            for batch, (filename, echo, displacement_vector, classifier) in enumerate(self.val_loader):
                classifier = classifier.reshape(-1,1)
                echo, displacement_vector, classifier = echo.to(self.device), displacement_vector.to(self.device), classifier.to(self.device)
                pred = self.model(echo)
                pred_displacement = pred[0]
                pred_classifier = pred[1]
                loss = self.loss_fn1(pred_displacement, displacement_vector) + self.loss_fn2(pred_classifier, classifier)
                loss /= len(echo)
                val_loss += loss.item() * len(echo)

                if batch % self.print_interval == 0:
                    loss_value, curr = loss.item(), batch*len(echo)
                    self.print(f'valid: {loss_value:.9e} [{curr:>3d}/{size:>3d}]')

        val_loss /= size
        self.end_time = time.time()
        self.print(f'valid: {val_loss:.9e} [average]')
        self.print(f'Time: {(self.end_time - self.start_time):>8f}\n')
        self.total_val_step += 1

        if self.total_val_step % self.log_interval == 0:
            self.logger.add_scalar(
                'validation loss', val_loss, self.total_val_step)

        return val_loss

    def start(self):
        self.print(f'Training on {self.device}...')
        self.start_time = time.time()
        start_epoch = 0
        if self.resume:
            start_epoch = self.ckpt['epoch']
            start_lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            self.print(
                f'Resume training from epoch {start_epoch+1} with LR={start_lr}...')

        for t in range(start_epoch, self.epochs):
            self.print('')
            self.print(
                f'Epoch {t+1} ({utils.current_time()})\n------------------------------')

            self.train()
            val_loss = self.eval()
            self.lr_scheduler.step(val_loss)
            self.logger.add_scalar(
                    'learning rate', self.optimizer.state_dict()['param_groups'][0]['lr'], t)

            if (t+1) % self.save_interval == 0:
                pth_file_path = os.path.join(self.pth_path, f'{str(t+1)}.pth')
                self.save_state(pth_file_path, t)
                self.print(f'Checkpoint saved to {pth_file_path}...')

            if val_loss <= self.last_val_loss:
                pth_file_path = os.path.join(
                    self.pth_path, f'{self.best_epoch}-best.pth')
                if os.path.exists(pth_file_path):
                    os.remove(pth_file_path)
                self.best_epoch = t+1
                self.last_val_loss = val_loss
                pth_file_path = os.path.join(
                    self.pth_path, f'{self.best_epoch}-best.pth')
                self.save_state(pth_file_path, t)
                self.print(f'Best checkpoint saved to {pth_file_path}...')
                

        pth_file_path = os.path.join(
            self.pth_path, f'{self.epochs}-latest.pth')
        self.save_state(pth_file_path, t)
        self.print(
            f'Completed {self.epochs} epochs; saved in "{self.pth_path}"')
        self.logger.close()

    def print(self, text):
        print(text)
        console_file = open(self.console_path, 'a+')
        console_file.write(text+'\n')
        console_file.close()

    def save_state(self, path, epoch):
        ckpt_dict = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
            'total_train_step': self.total_train_step,
            'total_val_step': self.total_val_step,
            'last_val_loss': self.last_val_loss,
            'best_epoch': self.best_epoch,
            'init_time': self.init_time,
        }
        torch.save(ckpt_dict, path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cardiac FCNN Trainer')
    parser.add_argument(
        '--config', type=str, 
        default='configs/reduceLROnPlateau.json', 
        help='configuration path')
    parser.add_argument('--ckpt', type=str,
                        default='pths/0/Plateau-0.5-10_2023-03-22-11-20-23/100.pth', 
                        help='resume checkpoint path')
    args = parser.parse_args()
    trainer = Trainer(utils.load_config(args.config), ckpt_path=args.ckpt)
    trainer.start()
