import torch
from torch.utils.data import DataLoader
from models.model import VisionModel
from data.dataset import VisionDataset
from utils.preprocessing import get_transform
import yaml

def train(config):
    # 모델 초기화
    model = VisionModel()
    
    # 데이터셋 로드
    transform = get_transform()
    train_dataset = VisionDataset(config['data_dir'], transform=transform)
    train_loader = DataLoader(train_dataset, 
                            batch_size=config['batch_size'],
                            shuffle=True)
    
    # 옵티마이저 설정
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = torch.nn.CrossEntropyLoss()
    
    # 학습 루프
    for epoch in range(config['epochs']):
        model.train()
        for batch_idx, data in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

if __name__ == "__main__":
    with open("configs/config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    train(config) 