import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.05):
        super(ContrastiveLoss, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.register_buffer("temperature", torch.tensor(temperature))
        self.verbose = False

    def forward(self, z_i, z_j):
        # --------------------------------------------
        # Define forward pass
        # --------------------------------------------
        batch_size = z_i.shape[0]
        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        if self.verbose:
            print("Similarity matrix\n", similarity_matrix, "\n")

        def l_ij(i, j):
            sim_i_j = similarity_matrix[i, j]
            if self.verbose:
                print(f"sim({i}, {j})={sim_i_j}")
            numerator = torch.exp(sim_i_j / self.temperature)
            one_for_not_i = torch.ones((2 * batch_size, )).scatter_(0, torch.tensor([i]), 0.0).to(z_i.device)
            if self.verbose:
                print(f"1{{k!={i}}}", one_for_not_i)
            denominator = torch.sum(
                one_for_not_i * torch.exp(similarity_matrix[i, :] / self.temperature)
            )
            if self.verbose:
                print("Denominator", denominator)
            loss_ij = -torch.log(numerator / denominator)
            if self.verbose:
                print(f"loss({i},{j})={loss_ij}\n")
            return loss_ij.squeeze(0)

        N = batch_size
        loss = 0.0
        for k in range(0, N):
            loss += l_ij(k, k + N) + l_ij(k + N, k)
        return 1.0 / (2*N) * loss


class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(SupervisedContrastiveLoss, self).__init__()
        self.register_buffer("temperature", torch.tensor(temperature))

    def forward(self, representations, labels, soft_labels=None, pseudo_weight=10):
        # --------------------------------------------
        # Define forward pass
        # --------------------------------------------
        batch_size = representations.shape[0]
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        loss = 0.0
        for i in range(batch_size):
            label_arr = (labels == labels[i]).clone().detach().float().scatter_(
                0, torch.tensor([i], device=representations.device), 0.0
            )
            if soft_labels is not None:
                soft_label_arr = torch.inner(soft_labels, soft_labels[i]).clone().detach()
                label_arr += pseudo_weight * soft_label_arr

            numerator = torch.exp(similarity_matrix[i, :] / self.temperature)
            one_for_not_i = torch.ones((batch_size, )).scatter_(0, torch.tensor([i]), 0.0).to(representations.device)
            denominator = torch.sum(
                one_for_not_i * torch.exp(similarity_matrix[i, :] / self.temperature)
            )
            loss_arr = -torch.log(numerator / denominator)
            loss += torch.sum(loss_arr * label_arr)
        return loss / batch_size


class KLWithSoftLabelLoss(nn.Module):
    def __init__(self, temperature, weight):
        super(KLWithSoftLabelLoss, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='batchmean')
        self.register_buffer("temperature", torch.tensor(temperature))
        self.register_buffer("loss_weight", torch.tensor(weight))

    def forward(self, logits, soft_target):
        loss = self.criterion(
            F.log_softmax(logits/self.temperature, dim=1), F.softmax(soft_target/self.temperature, dim=1)
        ) * (self.temperature * self.temperature * self.loss_weight)
        return loss
