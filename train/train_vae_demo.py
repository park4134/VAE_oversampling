import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter

import torch
import matplotlib.pyplot as plt
import torchvision.utils as vutils

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from models.vae import *
from utils import *

# MNIST 데이터 로딩
def load_mnist(batch_size=128):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Lambda(lambda x: x.view(-1))])
    
    train_dataset = datasets.MNIST(root=os.path.join(os.getcwd(), 'data', 'mnist'), train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root=os.path.join(os.getcwd(), 'data', 'mnist'), train=False, transform=transform, download=True)
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

# VAE의 손실 함수 (Reconstruction Loss + KL Divergence)
def vae_loss(reconstructed, original, mu, logvar):
    # Binary Cross Entropy Loss for reconstruction
    bce_loss = F.binary_cross_entropy(reconstructed, original, reduction='sum')
    
    # KL Divergence
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return bce_loss + kl_div

# Train 함수
def train_vae(model, train_loader, optimizer, device, epoch, writer=None):
    model.train()
    train_loss = 0

    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        reconstructed, mu, logvar = model(data)
        
        # Compute loss
        loss = vae_loss(reconstructed, data, mu, logvar)
        
        # Backward pass and optimization
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item() / len(data):.6f}')
        
        # TensorBoard에 손실값 기록
        if writer:
            writer.add_scalar('Loss/train', loss.item() / len(data), epoch * len(train_loader) + batch_idx)

    avg_loss = train_loss / len(train_loader.dataset)
    print(f'====> Epoch: {epoch} Average loss: {avg_loss:.4f}')
    return avg_loss

# Reconstructed 이미지 시각화 함수
def visualize_reconstruction(original, reconstructed, epoch, batch_idx, save_path):
    # 첫 번째 이미지의 배치를 시각화하기 위해 detach 후 CPU로 이동
    original = original.view(-1, 1, 28, 28).cpu().detach()
    reconstructed = reconstructed.view(-1, 1, 28, 28).cpu().detach()

    # 입력 이미지와 재구성된 이미지를 그리드 형태로 저장
    fig, axes = plt.subplots(1, 2)
    
    axes[0].imshow(vutils.make_grid(original, nrow=8, padding=2, normalize=True).permute(1, 2, 0))
    axes[0].set_title('Original Images')
    
    axes[1].imshow(vutils.make_grid(reconstructed, nrow=8, padding=2, normalize=True).permute(1, 2, 0))
    axes[1].set_title('Reconstructed Images')

    plt.savefig(os.path.join(save_path, f'reconstruction_epoch_{epoch}_batch_{batch_idx}.png'))
    plt.show()

# Test 함수에서 재구성 이미지 시각화 추가
def test_vae(model, test_loader, device, epoch, save_path, writer=None):
    model.eval()
    test_loss = 0

    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(test_loader):
            data = data.to(device)
            
            # Forward pass (reconstruction 및 latent variables 계산)
            reconstructed, mu, logvar = model(data)
            
            # 손실 계산
            loss = vae_loss(reconstructed, data, mu, logvar)
            test_loss += loss.item()

            # 재구성된 이미지 시각화 (첫 번째 배치만)
            if batch_idx == 0:
                visualize_reconstruction(data, reconstructed, epoch, batch_idx, save_path)

    avg_loss = test_loss / len(test_loader.dataset)
    print(f'====> Test set loss: {avg_loss:.4f}')

    # TensorBoard에 기록
    if writer:
        writer.add_scalar('Loss/test', avg_loss, epoch)
    
    return avg_loss


# Main 학습 및 테스트 실행 함수
def main():
    save_path = os.path.join(os.getcwd(), 'runs', 'vae_demo')
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    # Hyperparameters
    batch_size = 128
    epochs = 30
    learning_rate = 1e-4

    # Device 설정 (GPU 사용 가능 시)
    device = 'cuda:0'

    # Load data
    train_loader, test_loader = load_mnist(batch_size=batch_size)

    # VAE 모델 생성 (구현된 VAE 모델 불러오기)
    model = VAE(
        input_dim=784,
        hidden_dim=[512, 64],
        z_dim=10,
        device=device,
    )  # 이미 구현된 VAE 클래스
    model.to(device)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    # TensorBoard SummaryWriter 설정
    writer = SummaryWriter(log_dir=os.path.join(os.getcwd(), 'logs/vae_demo'))

    # Training loop
    for epoch in range(1, epochs + 1):
        train_vae(model, train_loader, optimizer, device, epoch, writer)
        test_vae(model, test_loader, device, epoch, save_path, writer)

    # 학습 완료 후 모델 저장
    torch.save(model.state_dict(), os.path.join(save_path, 'vae_mnist.pth'))

    # TensorBoard 종료
    writer.close()

if __name__ == '__main__':
    main()