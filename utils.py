import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from sklearn.model_selection import StratifiedKFold


class FiSHNN(nn.Module):
    """FiSHNN: shared EfficientNet-B0 first 5 layers + EfficientNet branches for species/habitat.
    
    The shared layers consist of the first 5 blocks of EfficientNet-B0 (Stem + Blocks 0-3).
    Each branch adds the remaining blocks (Blocks 4-7) and classifiers.
    """

    def __init__(self, num_species: int, num_habitats: int):
        super().__init__()
        
        # Create backbones with pre-trained weights
        spec_model = tv_models.efficientnet_b0(weights="DEFAULT")
        hab_model = tv_models.efficientnet_b0(weights="DEFAULT")
        
        # Shared layers: Stem + blocks 0 to 3 (indices 0-4 of features)
        # These are taken from spec_model and will be shared by both branches in forward pass
        self.shared = nn.Sequential(
            spec_model.features[0],
            spec_model.features[1],
            spec_model.features[2],
            spec_model.features[3],
            spec_model.features[4],
        )
        
        # Species branch: continue with remaining EfficientNet blocks (5-8)
        self.species_backbone = spec_model
        self.species_backbone.features = nn.Sequential(*list(spec_model.features.children())[5:])
        
        # Habitat branch: continue with remaining EfficientNet blocks (5-8)
        self.habitat_backbone = hab_model
        self.habitat_backbone.features = nn.Sequential(*list(hab_model.features.children())[5:])
        
        # Replace classifiers for task-specific outputs
        self.species_backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(1280, num_species),
        )
        self.habitat_backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(1280, num_habitats),
        )

    def forward(self, x):
        # Process through shared layers
        shared_out = self.shared(x)
        
        # Species branch: continue with remaining EfficientNet blocks
        s_feat = self.species_backbone.features(shared_out)
        s_feat = self.species_backbone.avgpool(s_feat)
        s_feat = torch.flatten(s_feat, 1)
        species_logits = self.species_backbone.classifier(s_feat)

        # Habitat branch: continue with remaining EfficientNet blocks
        h_feat = self.habitat_backbone.features(shared_out)
        h_feat = self.habitat_backbone.avgpool(h_feat)
        h_feat = torch.flatten(h_feat, 1)
        habitat_logits = self.habitat_backbone.classifier(h_feat)

        return species_logits, habitat_logits


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_weighted_loss(weights_dict, device):
    weights = torch.tensor([weights_dict[i] for i in range(len(weights_dict))], dtype=torch.float32, device=device)
    return nn.CrossEntropyLoss(weight=weights)


def grad_cam(model: nn.Module, images: torch.Tensor, target_class: int, branch: str = "species", target_layer: str = "auto"):
    """Compute Grad-CAM heatmaps for a batch of images.

    Args:
        model: Multi-task model (e.g., FiSHNN).
        images: Tensor of shape (B, 3, H, W) already on device.
        target_class: Class index to visualize.
        branch: "species" or "habitat".
        target_layer: Module name to hook for activations/gradients, or 'auto' to use last conv.
    Returns:
        np.ndarray heatmaps with shape (B, H, W) in [0, 1].
    """

    activations = {}
    gradients = {}

    def forward_hook(module, inp, out):
        activations["value"] = out.detach()

    def backward_hook(module, grad_in, grad_out):
        gradients["value"] = grad_out[0].detach()

    # Get target module directly from model structure
    if target_layer == 'auto' or '.' not in target_layer:
        # Use the last convolutional block in the specified branch
        # (With 5 shared stages, original indices 5-8 are in the branch. 
        # index 8 is the last conv layer, which is at position 3 in the branch)
        if branch == "species":
            target_module = model.species_backbone.features[3]
        else:
            target_module = model.habitat_backbone.features[3]
    else:
        # Try to get from named_modules first, fall back to direct access
        try:
            target_module = dict(model.named_modules())[target_layer]
        except KeyError:
            # Parse the layer name and access directly
            parts = target_layer.split('.')
            target_module = model
            for part in parts:
                if part.isdigit():
                    target_module = target_module[int(part)]
                else:
                    target_module = getattr(target_module, part)

    handle_fwd = target_module.register_forward_hook(forward_hook)
    handle_bwd = target_module.register_full_backward_hook(backward_hook)

    model.zero_grad()
    species_logits, habitat_logits = model(images)
    logits = species_logits if branch == "species" else habitat_logits
    selected = logits[:, target_class]
    selected.sum().backward()

    acts = activations["value"]  # (B, C, H, W)
    grads = gradients["value"]   # (B, C, H, W)

    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = (weights * acts).sum(dim=1)  # (B, H, W)
    cam = F.relu(cam)
    # Normalize per-sample
    cam_min = cam.flatten(1).min(dim=1)[0].view(-1, 1, 1)
    cam_max = cam.flatten(1).max(dim=1)[0].view(-1, 1, 1)
    cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

    handle_fwd.remove()
    handle_bwd.remove()
    return cam.detach().cpu().numpy()


def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.5):

    img = Image.open(img_path).convert("RGB")
    img_np = np.array(img)

    heatmap_uint8 = np.uint8(255 * heatmap)
    jet = mpl.colormaps["jet"]
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap_uint8]
    jet_img = Image.fromarray(np.uint8(jet_heatmap * 255)).resize((img_np.shape[1], img_np.shape[0]))
    jet_np = np.array(jet_img)

    superimposed = np.uint8(np.clip(alpha * jet_np + img_np, 0, 255))
    out = Image.fromarray(superimposed)

    os.makedirs(os.path.dirname(cam_path), exist_ok=True)
    out.save(cam_path)


def make_loader(data, species_y, habitat_y, indices, batch_size, shuffle=False):
    x = torch.tensor(data[indices]).permute(0, 3, 1, 2)
    ys = torch.tensor(species_y[indices], dtype=torch.long)
    yh = torch.tensor(habitat_y[indices], dtype=torch.long)
    ds = TensorDataset(x, ys, yh)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def accuracy_from_logits(logits, targets):
    return (logits.argmax(dim=1) == targets).float().mean().item()


def train_one_epoch(model, loader, opt, crit_species, crit_habitat, device):
    model.train()
    running_loss = 0.0
    running_loss_s = 0.0
    running_loss_h = 0.0
    acc_s, acc_h, count = 0.0, 0.0, 0
    for xb, ys, yh in loader:
        xb, ys, yh = xb.to(device), ys.to(device), yh.to(device)
        opt.zero_grad()
        species_logits, habitat_logits = model(xb)
        loss_s = crit_species(species_logits, ys)
        loss_h = crit_habitat(habitat_logits, yh)
        loss = loss_s + loss_h
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=5.0)
        opt.step()

        batch_size = xb.size(0)
        running_loss += loss.item() * batch_size
        running_loss_s += loss_s.item() * batch_size
        running_loss_h += loss_h.item() * batch_size
        acc_s += accuracy_from_logits(species_logits, ys) * batch_size
        acc_h += accuracy_from_logits(habitat_logits, yh) * batch_size
        count += batch_size
    return (
        running_loss / count,
        running_loss_s / count,
        running_loss_h / count,
        acc_s / count,
        acc_h / count,
    )


def evaluate(model, loader, crit_species, crit_habitat, device):
    model.eval()
    running_loss = 0.0
    running_loss_s = 0.0
    running_loss_h = 0.0
    acc_s, acc_h, count = 0.0, 0.0, 0
    all_species_logits = []
    all_habitat_logits = []
    all_species_targets = []
    all_habitat_targets = []
    with torch.no_grad():
        for xb, ys, yh in loader:
            xb, ys, yh = xb.to(device), ys.to(device), yh.to(device)
            species_logits, habitat_logits = model(xb)
            loss_s = crit_species(species_logits, ys)
            loss_h = crit_habitat(habitat_logits, yh)
            loss = loss_s + loss_h

            batch_size = xb.size(0)
            running_loss += loss.item() * batch_size
            running_loss_s += loss_s.item() * batch_size
            running_loss_h += loss_h.item() * batch_size
            acc_s += accuracy_from_logits(species_logits, ys) * batch_size
            acc_h += accuracy_from_logits(habitat_logits, yh) * batch_size
            count += batch_size

            all_species_logits.append(species_logits.detach().cpu())
            all_habitat_logits.append(habitat_logits.detach().cpu())
            all_species_targets.append(ys.cpu())
            all_habitat_targets.append(yh.cpu())

    species_logits_cat = torch.cat(all_species_logits)
    habitat_logits_cat = torch.cat(all_habitat_logits)
    species_targets_cat = torch.cat(all_species_targets)
    habitat_targets_cat = torch.cat(all_habitat_targets)

    return (
        running_loss / count,
        running_loss_s / count,
        running_loss_h / count,
        acc_s / count,
        acc_h / count,
        species_logits_cat,
        habitat_logits_cat,
        species_targets_cat,
        habitat_targets_cat,
    )


def make_single_loader(data, indices, labels, batch_size, shuffle=False):
    x = torch.tensor(data[indices]).permute(0, 3, 1, 2)
    y = torch.tensor(labels[indices], dtype=torch.long)
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_single_epoch(model, loader, opt, criterion, device):
    model.train()
    running_loss = 0.0
    acc, count = 0.0, 0
    for xb, y in loader:
        xb, y = xb.to(device), y.to(device)
        opt.zero_grad()
        logits = model(xb)
        loss = criterion(logits, y)
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=5.0)
        opt.step()

        batch_size = xb.size(0)
        running_loss += loss.item() * batch_size
        acc += accuracy_from_logits(logits, y) * batch_size
        count += batch_size
    return running_loss / count, acc / count


def evaluate_single(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    acc, count = 0.0, 0
    with torch.no_grad():
        for xb, y in loader:
            xb, y = xb.to(device), y.to(device)
            logits = model(xb)
            loss = criterion(logits, y)

            batch_size = xb.size(0)
            running_loss += loss.item() * batch_size
            acc += accuracy_from_logits(logits, y) * batch_size
            count += batch_size
    return running_loss / count, acc / count


def run_single_task_cv(task_name, data, labels, num_classes, weight_dict, device, epochs, init_lr, batch_size):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_acc = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(data, labels)):
        print(f"\n[Single {task_name}] Fold {fold + 1}/5")
        train_loader = make_single_loader(data, train_idx, labels, batch_size, shuffle=True)
        val_loader = make_single_loader(data, val_idx, labels, batch_size, shuffle=False)

        model = tv_models.efficientnet_b0(weights="DEFAULT")
        model.classifier = nn.Sequential(nn.Dropout(p=0.2, inplace=True), nn.Linear(1280, num_classes))
        model = model.to(device)

        opt = Adam(model.parameters(), lr=init_lr)
        criterion = make_weighted_loss(weight_dict, device)

        best_acc = 0.0
        for epoch in range(epochs):
            tr_loss, tr_acc = train_single_epoch(model, train_loader, opt, criterion, device)
            val_loss, val_acc = evaluate_single(model, val_loader, criterion, device)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch + 1:03d}/{epochs} | train_loss {tr_loss:.4f} val_loss {val_loss:.4f} | val_acc {val_acc:.3f}")

            if val_acc > best_acc:
                best_acc = val_acc

        fold_acc.append(best_acc)
    return np.array(fold_acc)