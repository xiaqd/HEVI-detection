from torch import nn
from torch.nn.modules.loss import _Loss

__all__ = ["JointLoss", "WeightedLoss"]


class WeightedLoss(_Loss):
    """Wrapper class around loss function that applies weighted with fixed factor.
    This class helps to balance multiple losses if they have different scales
    """

    def __init__(self, loss, weight=1.0):
        super().__init__()
        self.loss = loss
        self.weight = weight

    def forward(self, *input):
        return self.loss(*input) * self.weight


class JointLoss(_Loss):
    """
    Wrap two loss functions into one. This class computes a weighted sum of two losses.
    """

    def __init__(self, loss_list, loss_list_weight):
        super().__init__()
        self.loss_list = [WeightedLoss(l, lw) for l,lw in zip(loss_list, loss_list_weight)]


    def forward(self, *input):
        total_loss = 0.
        for l in self.loss_list:
            total_loss += l(*input)
        

        return total_loss