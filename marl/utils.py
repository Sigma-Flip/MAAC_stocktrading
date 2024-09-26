import torch
import torch.nn as nn


def categorical_sample(probs):
    """
    Sample an action from a categorical distribution defined by the given probabilities.

    Parameters:
        probs (torch.Tensor): A 1D tensor of probabilities for each action.

    Returns:
        actions (torch.Tensor): The sampled action indices.
    """
    actions = torch.multinomial(probs, 1)  # Sample from the categorical distribution
    return actions


def onehot_from_logits(logits):
    """
    Convert logits to one-hot encoded actions.

    Parameters:
        logits (torch.Tensor): A 1D tensor of action logits.

    Returns:
        onehot_actions (torch.Tensor): A one-hot encoded tensor of actions.
    """
    probs = nn.functional.softmax(logits, dim=0)  # Convert logits to probabilities
    actions = categorical_sample(probs)  # Sample actions from probabilities
    onehot_actions = torch.zeros_like(probs)  # Create a tensor for one-hot encoding
    onehot_actions[actions] = 1  # Set the sampled action to 1
    return onehot_actions


def hard_update(target, source):
    """
    Copy network parameters from source to target.

    Parameters:
        target (torch.nn.Module): Network to copy parameters to.
        source (torch.nn.Module): Network whose parameters to copy.
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def soft_update(target, source, tau):
    """
    Perform a soft update of target network parameters towards source network parameters.

    Parameters:
        target (torch.nn.Module): Network to copy parameters to.
        source (torch.nn.Module): Network whose parameters to copy.
        tau (float): Weight factor for the update (0 < tau < 1).
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def disable_gradients(module):
    """
    Disable gradients for all parameters in the module.

    Parameters:
        module (torch.nn.Module): The module to disable gradients for.
    """
    for p in module.parameters():
        p.requires_grad = False


def enable_gradients(module):
    """
    Enable gradients for all parameters in the module.

    Parameters:
        module (torch.nn.Module): The module to enable gradients for.
    """
    for p in module.parameters():
        p.requires_grad = True


# # Example Usage
# if __name__ == "__main__":
#     # Example probabilities for categorical sampling
#     probs = torch.tensor([0.2, 0.5, 0.3], dtype=torch.float32)
#     sampled_action = categorical_sample(probs)
#     print("Sampled Action Index:", sampled_action.item())
#
#     # Example logits for one-hot encoding
#     logits = torch.tensor([0.1, 1.0, 0.5], dtype=torch.float32)
#     onehot_actions = onehot_from_logits(logits)
#     print("One-Hot Encoded Actions:", onehot_actions)
#
#
#     # Example networks for hard and soft updates
#     class SimpleNN(nn.Module):
#         def __init__(self):
#             super(SimpleNN, self).__init__()
#             self.fc = nn.Linear(10, 5)
#
#         def forward(self, x):
#             return self.fc(x)
#
#
#     target_net = SimpleNN()
#     source_net = SimpleNN()
#
#     # Hard update
#     hard_update(target_net, source_net)
#
#     # Soft update
#     tau = 0.1
#     soft_update(target_net, source_net, tau)
#
#     print("Updated Target Network Parameters:", list(target_net.parameters()))
