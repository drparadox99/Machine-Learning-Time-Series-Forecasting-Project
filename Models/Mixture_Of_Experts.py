import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Expert, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.network(x)


class GatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts):
        super(GatingNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, num_experts),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.network(x)


class MixtureOfExperts(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_experts):
        super(MixtureOfExperts, self).__init__()
        self.experts = nn.ModuleList([Expert(input_dim, hidden_dim, output_dim) for _ in range(num_experts)])
        self.gating_network = GatingNetwork(input_dim, num_experts)

    def forward(self, x):
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        gating_weights = self.gating_network(x).unsqueeze(2)
        output = torch.sum(gating_weights * expert_outputs, dim=1)
        return output


# Example usage:
input_dim = 10  # Number of features
hidden_dim = 50
output_dim = 1  # Forecast horizon
num_experts = 3

model = MixtureOfExperts(input_dim, hidden_dim, output_dim, num_experts)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Example dummy input and target
x = torch.randn(64, input_dim)  # Batch size of 64
y = torch.randn(64, output_dim)

# Training step
optimizer.zero_grad()
output = model(x)
loss = F.mse_loss(output, y)
loss.backward()
optimizer.step()

print("Output shape:", output.shape)
