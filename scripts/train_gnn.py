import numpy as np
import random

import os
import gc
import copy

from tqdm import tqdm

import torch
import torch.nn as nn
from torch_geometric.data import InMemoryDataset, Data

import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing, global_mean_pool, global_max_pool

import matplotlib.pyplot as plt

from timm.timm.data.dataset_factory import create_dataset
from timm.timm.models import create_model

seed = 42  # You can use any number

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # If using multi-GPU
np.random.seed(seed)
random.seed(seed)

# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def load_model(model_path):
    model = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
    arch = create_model(model['arch'])
    arch.load_state_dict(model['state_dict'])
    return arch.eval()


def feature_collect(model: nn.Module, images: torch.Tensor, collect_input: bool = True):
    tracked_layers = (nn.Conv2d, nn.Linear)
    feature_dict = {}

    def feature_hook(name):
        def hook(module, f_in, f_out):
            idx = len(feature_dict)  # Maintain layer index
            if collect_input:
                feature_dict[(idx, name)] = f_in[0].detach().cpu() if isinstance(f_in, tuple) else f_in.detach().cpu()
            else:
                feature_dict[(idx, name)] = f_out.detach().cpu()

        return hook

    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, tracked_layers):
            hooks.append(module.register_forward_hook(feature_hook(name)))

    output = model(images)

    for hook in hooks:
        hook.remove()  # Cleanup hooks

    return feature_dict, output


def compute_neural_act(features_hook):
    neural_act = []
    for k in features_hook:
        if len(features_hook[k][0].shape) == 3:
            layer_act = [features_hook[k][i].max(1)[0].max(1)[0].unsqueeze(1) for i in range(len(features_hook[k]))]
        else:
            layer_act = [features_hook[k][i].unsqueeze(1) for i in range(len(features_hook[k]))]

        layer_act = torch.cat(layer_act, dim=1)
        # layer_act=torch.cat(layer_act, dim=1).amax(dim=0, keepdim=True)
        # layer_act=(layer_act-layer_act.mean())/(layer_act.std())
        layer_act = (layer_act - layer_act.mean(1, keepdim=True)) / (layer_act.std(1, keepdim=True) + 1e-30)

        neural_act.append(layer_act)
    neural_act = torch.cat(neural_act)
    return neural_act


def mat_discorr_adjacency(X: torch.tensor, Y: torch.tensor = None) -> torch.tensor:
    """
    Distance-correlation matrix calculation in tensor format. Return pairwise distance correlation among all row vectors in X.

    Dist-corr between two vector a and b is:

        dist-corr(a, b) = (1/d^2)\sum_{i=1}^d \sum_{j=1}^d A_{i,j} B_{i, j}

        where:
            A_{i,j} = a_{i,j} - a_{i, .} - a_{., j} + a_{., .}
            B_{i,j} = b_{i,j} - b_{i, .} - b_{., j} + b_{., .}

            a_{i,j} = |a_i - a_j|_p
            b_{i,j} = |b_i - b_j|_p
            a_{i, .} = (1/d) sum_{j=1}^d a_{i, j}
            a_{., j} = (1/d) sum_{i=1}^d a_{i, j}
            a_{., .} = (1/d^2) sum_{i=1}^d sum_{j=1}^d a_{i, j}

    Input args:
        X (torch.tensor). n*d. n is the number of neurons and d is the feature dimension.
        Y (torch.tensor). Optional.
    """
    n, m = X.shape
    # If Y is not given, then calculate distcorr(X, X)
    if not Y:
        Y = X
    # Constrain the size of tensor to be sent to GPU to avoid memory overflow
    # Con-comment to use GPU
    # if (64*n**2)/(10**9) < 8:
    #     X = X.cuda()
    #     Y = Y.cuda()
    bpd = torch.cdist(X.unsqueeze(2), Y.unsqueeze(2), p=2)
    bpd = bpd - bpd.mean(axis=1)[:, None, :] - bpd.mean(axis=2)[:, :, None] + bpd.mean((1, 2))[:, None, None]
    pd = torch.mm(bpd.view(n, -1), bpd.view(n, -1).T)
    del bpd, X, Y
    gc.collect()
    torch.cuda.empty_cache()

    pd /= n ** 2
    pd = torch.sqrt(pd)
    pd /= (torch.sqrt(torch.diagonal(pd)[None, :] * torch.diagonal(pd)[:, None]) + 1e-8)
    pd.fill_diagonal_(1)

    return pd


def distance_matrix(model, images):
    features_hook, output_hook = feature_collect(model, images)
    neural_act = compute_neural_act(features_hook)
    discorr_mat = mat_discorr_adjacency(neural_act)
    D = 1 - discorr_mat.detach().cpu().numpy()

    return D


def image_to_trigger_dataset(image, patch_size=2, step_size=2, stim_level=4):
    """
    return #(shape/pos_w) * (shape/pos_h), #stim_level, *image.size
    """
    prob_inputs = []
    input_valuerange = [0, 255]
    stim_seq = np.linspace(input_valuerange[0], input_valuerange[1], stim_level)
    input_shape = image.shape
    input_eg = copy.deepcopy(image)
    for pos_w in range(0, input_shape[1] - patch_size + 1, step_size):
        for pos_h in range(0, input_shape[2] - patch_size + 1, step_size):
            count = 0
            prob_input = input_eg.repeat(len(stim_seq), 1, 1, 1)
            for i in stim_seq:
                prob_input[count, :,
                int(pos_w):min(int(pos_w + patch_size), input_shape[1]),
                int(pos_h):min(int(pos_h + patch_size), input_shape[1])] = i
                count += 1
            prob_inputs.append(prob_input)
    return torch.stack(prob_inputs).float()




dataset = create_dataset(
    f"torch/image_folder",
    root="experiments/cifar10",
    split="valid",
    is_training=True,
)

image, _ = dataset[1000]
image = np.asarray(image)
image = torch.from_numpy(image.astype(np.float32))
image = image.permute((2, 0, 1))

trigger_dataset = image_to_trigger_dataset(image, patch_size=2, step_size=30, stim_level=4)

X = []
y = []
for model_name in tqdm(os.listdir("experiments/models_cleaned/")):
    clean_model = load_model(f"experiments/models_cleaned/{model_name}")
    trigger_model = load_model(f"experiments/models_triggered/{model_name}")
    for j, model in enumerate([clean_model, trigger_model]):
        X.append([])
        y.append(j)
        for batch in trigger_dataset:
            d = distance_matrix(model, batch)
            X[-1].append(d)

        X[-1] = np.stack(X[-1], axis=0)


class DistanceMatrixInMemoryDataset(InMemoryDataset):
    def __init__(self, distance_matrices, labels, transform=None, pre_transform=None):
        self.distance_matrices = distance_matrices
        self.labels = labels
        super().__init__(None, transform, pre_transform)

        # Convert all distance matrices to Data objects
        data_list = self.process_data()

        # Store the dataset in-memory
        data, slices = self.collate(data_list)
        self.data = data
        self.slices = slices

    @property
    def num_node_features(self):
        return 1

    @property
    def num_classes(self):
        return len(np.unique(self.labels))

    def process_data(self):
        """Processes all samples into PyG Data objects."""
        data_list = []

        for i in range(len(self.distance_matrices)):
            matrices = self.distance_matrices[i]
            label = self.labels[i]

            n = matrices[0].shape[0]
            x_coords, y_coords = np.meshgrid(np.arange(n), np.arange(n))
            x = x_coords.flatten()
            y = y_coords.flatten()

            edge_index = np.array([x, y])
            edge_attr = 1 - np.stack(matrices)[:, x, y].T

            # Filter edges
            filt = np.all((edge_attr < 0.8) & (edge_attr != 0), axis=1)
            edge_index = edge_index[:, filt]
            edge_attr = edge_attr[filt]

            # Relabel node indices
            unique_nodes, new_indices = np.unique(edge_index, return_inverse=True)
            edge_index = new_indices.reshape(2, -1)

            # Convert to torch tensors
            edge_index = torch.tensor(edge_index, dtype=torch.long)
            edge_attr = torch.tensor(edge_attr, dtype=torch.float)

            edge_attr = (edge_attr - edge_attr.mean(axis=0)) / (edge_attr.std(axis=0))

            # Node features (1D feature vector for remaining nodes)
            # x = torch.ones((matrices.shape[1], 1))
            x = torch.ones((len(unique_nodes), 1))

            # Create Data object
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=torch.tensor([label], dtype=torch.long))
            data_list.append(data)

        return data_list


dataset = DistanceMatrixInMemoryDataset(X, y)

train_dataset = dataset[:100]
valid_dataset = dataset[100:130]
test_dataset = dataset[130:]

print(f'Number of training graphs: {len(train_dataset)}')
print(f'Number of valid graphs: {len(valid_dataset)}')
print(f'Number of test graphs: {len(test_dataset)}')


train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False)

for step, data in enumerate(train_loader):
    print(f'Step {step + 1}:')
    print('=======')
    print(f'Number of graphs in the current batch: {data.num_graphs, data.batch.max()}')
    print(data)
    print()

for step, data in enumerate(valid_loader):
    print(f'Step {step + 1}:')
    print('=======')
    print(f'Number of graphs in the current batch: {data.num_graphs}')
    print(data)
    print()


# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# === Define Custom Message-Passing Layer with Edge Features ===
class EdgeGNN(MessagePassing):
    def __init__(self, in_channels, edge_channels, hidden_dim):
        super(EdgeGNN, self).__init__(aggr="mean")

        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_channels, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(in_channels + hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.edge_mlp(edge_attr)
        return self.propagate(edge_index, x=x, edge_attr=edge_embedding)

    def message(self, x_j, edge_attr):
        return torch.cat([x_j, edge_attr], dim=1)

    def update(self, aggr_out):
        return self.dropout(self.node_mlp(aggr_out))


# === Define GNN Model for Graph Classification ===
class FullyConnectedGNN(nn.Module):
    def __init__(self, in_channels, edge_channels, hidden_dim, num_classes):
        super(FullyConnectedGNN, self).__init__()
        self.conv1 = EdgeGNN(in_channels, edge_channels, hidden_dim)
        self.conv2 = EdgeGNN(hidden_dim, edge_channels, hidden_dim)
        self.conv3 = EdgeGNN(hidden_dim, edge_channels, hidden_dim)
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.conv1(x, edge_index, edge_attr)
        x = self.conv2(x, edge_index, edge_attr)
        x = self.conv3(x, edge_index, edge_attr)
        x = global_mean_pool(x, batch)
        return self.fc_out(x)


# === Weight Initialization Function ===
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm1d):  # Initialize BatchNorm layers
        nn.init.ones_(m.weight)  # Scale initialized to 1
        nn.init.zeros_(m.bias)  # Bias initialized to 0


# === Training Function with AMP and Gradient Clipping ===
def train(model, loader, optimizer, criterion, scaler, device):
    model.train()
    total_loss, correct = 0, 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
            loss = criterion(out, data.y)

        scaler.scale(loss).backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        correct += (out.argmax(dim=1) == data.y).sum().item()

    return total_loss / len(loader), correct / len(loader.dataset)


# === Evaluation Function ===
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
            loss = criterion(out, data.y)
            total_loss += loss.item()
            correct += (out.argmax(dim=1) == data.y).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)


# === Initialize Model, Optimizer, Scheduler, and Loss ===
num_features = 1
num_edge_features = 4
num_classes = 2
hidden_dim = 32

# === Train the Model with Early Stopping ===
num_epochs = 500
patience = 20
best_loss = float("inf")
counter = 0

model = FullyConnectedGNN(in_channels=num_features, edge_channels=num_edge_features, hidden_dim=hidden_dim,
                          num_classes=num_classes).to(device)

optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                       T_max=100)  # ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)  # Use ReduceLROnPlateau
criterion = nn.CrossEntropyLoss()
scaler = torch.cuda.amp.GradScaler()

train_losses = []
valid_losses = []

for epoch in range(1, num_epochs + 1):
    train_loss, train_acc = train(model, train_loader, optimizer, criterion, scaler, device)
    valid_loss, valid_acc = evaluate(model, valid_loader, criterion, device)
    scheduler.step(valid_loss)  # Use the scheduler step based on test loss

    train_losses.append(train_loss)
    valid_losses.append(valid_loss)

    print(
        f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Test Loss: {valid_loss:.4f}, Test Acc: {valid_acc:.4f}")

    if valid_loss < best_loss:
        best_loss = valid_loss
        counter = 0
        torch.save(model.state_dict(), "best_model.pth")
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

# Plot the loss curves
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
plt.plot(range(1, len(valid_losses) + 1), valid_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

print("Model training complete and saved.")

model.load_state_dict(torch.load("best_model.pth"))

acc = []
for i in tqdm(range(100)):
    test_loader = DataLoader(test_dataset[np.random.randint(0, len(test_dataset), 30)], batch_size=16, shuffle=False)
    loss, accuracy = evaluate(model, test_loader, criterion, device)
    acc.append(accuracy)
acc = np.array(acc)
acc.mean(), acc.std()

print(f"mean +/- std: {acc.mean()} +/- {acc.std()}")