from torch import nn


class Maplinear(nn.Module):
    """Classification Head for  transformer encoders"""

    def __init__(self, class_size, embed_size):
        super().__init__()
        self.class_size = class_size
        self.embed_size = embed_size
        # self.mlp1 = nn.Linear(embed_size, embed_size)
        # self.mlp2 = (nn.Linear(embed_size, class_size))
        self.maplinear = nn.Linear(class_size, embed_size)

    def forward(self, hidden_state):
        # hidden_state = nn.functional.relu(self.mlp1(hidden_state))
        # hidden_state = self.mlp2(hidden_state)
        logits = self.maplinear(hidden_state)
        return logits
