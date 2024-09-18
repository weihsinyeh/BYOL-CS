import os
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import logging
import matplotlib.pyplot as plt
logger = logging.getLogger(__name__)

class BYOLTrainer:
    def __init__(self, online_network, target_network, predictor, optimizer, device, **params):
        self.online_network = online_network
        self.target_network = target_network
        self.optimizer = optimizer
        self.summary_writer = SummaryWriter(os.path.join(params['logdir'], 'train_logs'))
        self.device = device
        self.predictor = predictor

        # trainer
        self.batch_size = params['batch_size']
        self.m = params['m']
        self.checkpoint_interval = params['checkpoint_interval']
        self.max_epochs = params['max_epochs']
        self.checkpoint_dir = params['checkpointdir']

    @torch.no_grad()
    def _update_target_network_parameters(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @staticmethod
    def regression_loss(x, y):
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)
        return 2 - 2 * (x * y).sum(dim=-1)

    def initializes_target_network(self):
        # init momentum network as encoder net
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

    def train(self, train_dataset):
        niter = 0
        model_checkpoints_folder = os.path.join(self.checkpoint_dir, 'checkpoints')

        self.initializes_target_network()
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size,
                                  num_workers=1, drop_last=False, shuffle=True)
        for epoch in range(self.max_epochs):
            loss_list = []
            print("Epoch :",epoch)
            for (batch_view_1, batch_view_2), _ in tqdm(train_loader):
                batch_view_1 = batch_view_1.to(self.device)
                batch_view_2 = batch_view_2.to(self.device)

                '''
                image1 = batch_view_1[8].permute(1, 2, 0).cpu().numpy()
                image2 = batch_view_2[8].permute(1, 2, 0).cpu().numpy()
    
                fig, axs = plt.subplots(1, 2, figsize=(10, 5)) 
                axs[0].imshow(image1)
                axs[0].axis('off')  
                axs[0].set_title('Image 1')  

                axs[1].imshow(image2)
                axs[1].axis('off')  
                axs[1].set_title('Image 2')  
                plt.tight_layout()  
                plt.show()
                '''

                loss = self.update(batch_view_1, batch_view_2)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self._update_target_network_parameters()  # update the key encoder
                niter += 1
                loss_list.append(loss.item())

            if epoch % 1 == 0:
                checkpoint_name = 'model_epoch' + str(epoch) + '.pth'
                self.save_model(os.path.join(model_checkpoints_folder, checkpoint_name))
            self.summary_writer.add_scalar('train/loss', np.mean(loss_list), epoch)
            logger.info(f"Epoch {epoch} : Loss {np.mean(loss_list)}")
            print(f"Epoch {epoch} : Loss {np.mean(loss_list)}")


    def update(self, batch_view_1, batch_view_2):
        # compute query feature
        predictions_from_view_1 = self.predictor(self.online_network(batch_view_1))
        predictions_from_view_2 = self.predictor(self.online_network(batch_view_2))

        # compute key features
        with torch.no_grad():
            targets_to_view_2 = self.target_network(batch_view_1)
            targets_to_view_1 = self.target_network(batch_view_2)

        loss = self.regression_loss(predictions_from_view_1, targets_to_view_1)
        loss += self.regression_loss(predictions_from_view_2, targets_to_view_2)
        return loss.mean()

    def save_model(self, PATH):

        torch.save({
            'online_network_state_dict': self.online_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, PATH)