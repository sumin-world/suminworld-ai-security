# attacks/fgsm_mnist_starter.py
import os, math, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1) 아주 작은 CNN (LeNet 유사)
class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 16, 3, 1)      # 28->26
        self.c2 = nn.Conv2d(16, 32, 3, 1)     # 26->24
        self.fc1 = nn.Linear(32*24*24, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.c1(x))
        x = F.relu(self.c2(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# 2) 데이터
def get_loaders(batch=128):
    tfm = transforms.ToTensor()
    train = datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
    test  = datasets.MNIST(root="./data", train=False, download=True, transform=tfm)
    return (torch.utils.data.DataLoader(train, batch_size=batch, shuffle=True),
            torch.utils.data.DataLoader(test,  batch_size=batch, shuffle=False))

# 3) 학습
def train_one_epoch(model, loader, opt, loss_fn):
    model.train()
    total, correct, run_loss = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()
        run_loss += loss.item() * x.size(0)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += x.size(0)
    return correct/total, run_loss/total

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total, correct = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += x.size(0)
    return correct/total

# 4) FGSM (sign(∇x loss) 이용)
def fgsm_attack(images, labels, model, eps=0.1):
    model.eval()
    images = images.clone().detach().to(device)
    labels = labels.to(device)
    images.requires_grad_(True)
    logits = model(images)
    loss = F.cross_entropy(logits, labels)
    model.zero_grad()
    loss.backward()
    # 입력 방향으로 작은 한 스텝
    adv_images = images + eps * images.grad.detach().sign()
    # [0,1] 클램프
    adv_images = torch.clamp(adv_images, 0, 1)
    return adv_images

def evaluate_under_attack(model, loader, eps=0.1, save_examples=8):
    model.eval()
    total, correct = 0, 0
    saved = 0
    os.makedirs("outputs", exist_ok=True)
    for x, y in loader:
        x_adv = fgsm_attack(x, y, model, eps=eps)
        logits = model(x_adv.to(device))
        pred = logits.argmax(1).detach().cpu()
        correct += (pred == y).sum().item()
        total += x.size(0)
        # 샘플 저장
        if saved < save_examples:
            grid = torch.cat([x[:save_examples], x_adv[:save_examples]], dim=0)
            save_image(grid, f"outputs/fgsm_eps{eps}_examples.png")
            saved = save_examples
    return correct/total

def main():
    start = time.time()
    train_loader, test_loader = get_loaders(batch=128)
    model = TinyCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    print("[*] Training 1 epoch (quick)…")
    acc_tr, loss_tr = train_one_epoch(model, train_loader, opt, loss_fn)
    acc_te = evaluate(model, test_loader)
    print(f" - train acc: {acc_tr*100:.2f}%, test acc: {acc_te*100:.2f}%")

    eps = 0.10
    print(f"[*] Evaluating under FGSM attack, eps={eps} …")
    acc_adv = evaluate_under_attack(model, test_loader, eps=eps)
    print(f" - adv acc: {acc_adv*100:.2f}% (↓ {max(0.0, (acc_te-acc_adv)*100):.2f}%)")

    print(f"[*] Saved examples to outputs/fgsm_eps{eps}_examples.png")
    print(f"Done in {time.time()-start:.1f}s")

if __name__ == "__main__":
    main()
