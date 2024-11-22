import math
import os
from datetime import datetime
import random
import numpy as np
import torch
from torch import optim
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from dataset.data_loader_deap import get_TensorDataset_deap
from dataset.data_loader_seed import get_TensorDataset_seed
from metrics.KNN_Deap2 import knn_Deap
from metrics.KNN_seed import knn_Seed
from models.Discriminator import train_discriminator, Discriminator_Channel_32, Discriminator_Channel_62
from models.Generator import EEGfuseNet_SEED, EEGfuseNet_DEAP
from models.MI_model import MIFCNet
from utils.utils import weigth_init


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class EMALoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, running_ema):
        ctx.save_for_backward(input, running_ema)
        input_log_sum_exp = input.exp().mean().log()

        return input_log_sum_exp

    @staticmethod
    def backward(ctx, grad_output):
        input, running_mean = ctx.saved_tensors
        grad = grad_output * input.exp().detach() / \
               (running_mean + 1e-6) / input.shape[0]
        return grad, None


def ema(mu, alpha, past_ema):
    return alpha * mu + (1.0 - alpha) * past_ema


def ema_loss(x, running_mean, alpha):
    t_exp = torch.exp(torch.logsumexp(x, 0) - math.log(x.shape[0])).detach()
    if running_mean == 0:
        running_mean = t_exp
    else:
        running_mean = ema(t_exp, alpha, running_mean.item())
    t_log = EMALoss.apply(x, running_mean)

    # Recalculate ema

    return t_log, running_mean


def compute_mi_loss(scores1, scores2, epsilon=1e-10):
    # Calculate mutual information loss
    mi_loss = -(torch.sum(torch.log(scores1 + epsilon)) + torch.sum(torch.log(1 - scores2 + epsilon))) / (
            scores1.size(0) + scores2.size(0))
    return mi_loss


def donsker_varadhan_loss(pos_scores, neg_scores):
    loss = 1
    return loss


def fenchel_dual_loss(pos_scores, neg_scores):
    loss = 1
    return loss


def infonce_loss(pos_scores, neg_scores):
    loss = 1
    return loss


def mine_loss(pos_scores, neg_scores, running_mean, alpha):
    neg_scores, running_mean = ema_loss(neg_scores, running_mean, alpha)
    mi_loss = -pos_scores.mean() + neg_scores
    return mi_loss, running_mean


def fdiv_loss(pos_scores, neg_scores):
    neg_scores = torch.exp(neg_scores - 1).mean()
    mi_loss = -pos_scores.mean() + neg_scores
    return mi_loss


def mine_biased_loss(pos_scores, neg_scores):
    neg_scores = torch.logsumexp(
        neg_scores, 0) - math.log(neg_scores.shape[0])
    mi_loss = -pos_scores.mean() + neg_scores
    return mi_loss


def mine2_loss(pos_scores, neg_scores, ma_et):
    neg_scores = torch.exp(neg_scores)
    ma_rate = 0.01
    ma_et = (1 - ma_rate) * ma_et + ma_rate * torch.mean(neg_scores)  # 更新 et 的指数化边缘分布的移动平均值。
    loss = -(torch.mean(pos_scores) - (1 / ma_et.mean()).detach() * torch.mean(
        neg_scores))  # 这里使用了未偏估unbiased estimator）来减少估计的方差。
    return loss, ma_et


def learn_mi(feature_map, latent_code, mi_model, losstype, running_mean, ma_et):
    alpha = 0.01
    feature_map = feature_map.to(device)
    latent_code = latent_code.to(device)
    latent_code_shuffle = latent_code[torch.randperm(latent_code.size(0))]
    pos = torch.cat((feature_map, latent_code), dim=1)
    neg = torch.cat((feature_map, latent_code_shuffle), dim=1)
    pos_scores = mi_model(pos)  # Shape: (128, 1)
    neg_scores = mi_model(neg)  # Shape: (128, 1)
    # Reshape scores to 1-dimensional tensors
    pos_scores = pos_scores.squeeze()  # Shape: (128,)
    neg_scores = neg_scores.squeeze()  # Shape: (128,)
    if losstype == 'mine':
        mi_loss, running_mean = mine_loss(pos_scores, neg_scores, running_mean, alpha)
    elif losstype == 'fdiv':
        mi_loss = fdiv_loss(pos_scores, neg_scores)
    elif losstype == 'mine_biased':
        mi_loss = mine_biased_loss(pos_scores, neg_scores)
    elif losstype == 'mine2':
        mi_loss, ma_et = mine2_loss(pos_scores, neg_scores, ma_et)
    if losstype == 'fd':
        mi_loss = fenchel_dual_loss(pos_scores, neg_scores)
    elif losstype == 'nce':
        mi_loss = infonce_loss(pos_scores, neg_scores)
    elif losstype == 'dv':
        mi_loss = donsker_varadhan_loss(pos_scores, neg_scores)
    # else:
    #     mi_loss = compute_mi_loss(pos_scores, neg_scores)
    # mi_loss = -pos_scores + neg_scores
    # Compute the mutual information loss
    return mi_loss, running_mean, ma_et


def model_validation_autoencoder(model, loader_valid):
    model.eval()
    loss = 0
    loss_function = torch.nn.MSELoss(reduction='mean')
    with torch.no_grad():
        for step, (data) in enumerate(loader_valid):
            inputs = Variable(data[0].to(device))
            latent_code, feature_map = model(inputs)
            output = model.decoder(latent_code)
            loss += loss_function(inputs, output)
    loss_mean = loss / (step + 1)
    return loss_mean


def pretrain_model_g(dataset, loader_valid, Generator, mi_model, Discriminator, LR, EPOCH, loss_type):
    opti_gen = optim.Adam(Generator.parameters(), lr=LR)
    # opti_mi = optim.Adam(mi_model.parameters(), lr=LR / 5)
    opti_dis = optim.Adam(Discriminator.parameters(), lr=LR / 5)
    loss_train = np.zeros(EPOCH)
    loss_valid = np.zeros(EPOCH)
    loss_function = torch.nn.MSELoss(reduction='mean')
    # for epoch in range(EPOCH):
    for epoch in range(EPOCH):
        print(f"epoch {epoch}")
        Generator.train()
        recon_epoch_loss = 0
        # mi_epoch_loss = 0
        dis_epoch_loss = 0
        running_mean = 0
        ma_et = 0
        for step, (batch_x) in enumerate(dataset):
            inputs = Variable(batch_x[0].to(device))
            dis_loss = train_discriminator(inputs, Discriminator, Generator, opti_dis, lamda=1)
            Generator.train()
            # opti_mi.zero_grad()
            opti_gen.zero_grad()
            latent_code, feature_map, feature_map_lsit = Generator(inputs)
            latent_code = latent_code.to(device)
            feature_map = feature_map.to(device)
            recon = Generator.decoder(latent_code)
            recon_loss = loss_function(inputs, recon)
            mi_loss, running_mean, ma_et = learn_mi(feature_map, latent_code, mi_model, loss_type, running_mean, ma_et)
            # feature_map:128*192,latent_code:128*192
            # loss = mi_loss + recon_loss
            loss = recon_loss
            loss.backward()
            # opti_mi.step()
            opti_gen.step()
            with torch.no_grad():
                recon_epoch_loss += recon_loss.data
                # mi_epoch_loss += mi_loss.data
                dis_epoch_loss += dis_loss
            torch.cuda.empty_cache()
        loss_train[epoch] = loss / (step + 1)
        loss_valid[epoch] = model_validation_autoencoder(Generator, loader_valid)
        # print("reconloss:{}----------MIloss{}".format(recon_loss, mi_loss))
        log_dir = os.path.join('../logs', f'{norm_type}_MIlog')
        with SummaryWriter(log_dir=log_dir,
                           comment='train') as writer:  # 可以直接使用python的with语法，自动调用close方法
            # 损失函数图像
            writer.add_scalar('recon_loss', recon_epoch_loss / (step + 1), epoch + 1)
            writer.add_scalar('dis_loss', dis_epoch_loss / (step + 1), epoch + 1)
            # writer.add_scalar('mi_loss', mi_epoch_loss / (step + 1), epoch + 1)
        print(f'EPOCH_pre {epoch:03d}: Training Loss: {loss_train[epoch]:.4f}')
        print(f'EPOCH_pre {epoch:03d}: Validation Loss: {loss_valid[epoch]:.4f}')
    return Generator


def train_Autoencoder_g(sub, model_id, norm_type, dataset_type, loss_type):
    if dataset_type == 'seed':
        loader_train, loader_valid = get_TensorDataset_seed(sub)
        Generator = EEGfuseNet_SEED(16, 1, 1, 384).to(device)
        Discriminator = Discriminator_Channel_62(1, 1, 384).to(device)
    elif dataset_type == 'deap':
        loader_train, loader_valid = get_TensorDataset_deap(sub)
        Generator = EEGfuseNet_DEAP(16, 1, 1, 384).to(device)
        Discriminator = Discriminator_Channel_32(1, 1, 384).to(device)
    mi_model = MIFCNet(384, 192).to(device)
    Generator.apply(weigth_init)
    Discriminator.apply(weigth_init)
    mi_model.apply(weigth_init)
    Generator = pretrain_model_g(loader_train, loader_valid, Generator, mi_model, Discriminator, 0.001, 100, loss_type)
    if model_id == 'RNN':
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        path = os.path.join('../checkpoints',
                            f'MSE+Dis_{dataset_type}_{norm_type}_sub{sub}_session1_{now}.pkl')
        torch.save(Generator.state_dict(), path)


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train = False
    # train = True
    if train:
        loss_types = ['mine2']
        # loss_type = ['mine', 'fdiv', 'mine_biased', 'mine2']
        norm_types = ['global_scale_value', 'global_gaussian_value']
        for norm_type in norm_types:
            for loss_type in loss_types:
                train_Autoencoder_g(0, 'RNN', norm_type, 'deap', loss_type)
    # result = knn_Seed()
    setup_seed(3)
    result = knn_Deap()
    args = {'hidden_dim': 16, 'n_layer': 1, 'n_filters': 1, 'input_size': 384, 'dataset_type': 'seed', 'norm_type': '',
            'batch_size': 48, 'device': 'cuda' if torch.cuda.is_available() else 'cpu', 'alpha': 0.5}