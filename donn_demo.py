import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift, ifftshift
import warnings
warnings.filterwarnings('ignore')

# ==================== 1. 衍射传播物理模型 ====================
class DiffractivePropagation(nn.Module):
    """
    实现自由空间中的角谱衍射传播
    基于 Rayleigh-Sommerfeld 衍射理论
    """
    def __init__(self, wavelength=0.75e-3, pixel_size=0.04, distance=40e-3):
        """
        参数:
            wavelength: 波长 (mm)，太赫兹波段典型值
            pixel_size: 像素尺寸 (mm)
            distance: 传播距离 (mm)
        """
        super().__init__()
        self.wavelength = wavelength
        self.pixel_size = pixel_size
        self.distance = distance
        self.k = 2 * np.pi / wavelength  # 波数
        
    def forward(self, input_field):
        """
        输入: input_field [batch, height, width] 复数光场
        输出: propagated_field [batch, height, width] 传播后的复数光场
        """
        batch_size, H, W = input_field.shape
        
        # 创建空间频率网格
        fx = np.fft.fftfreq(W, d=self.pixel_size)
        fy = np.fft.fftfreq(H, d=self.pixel_size)
        FX, FY = np.meshgrid(fx, fy)
        
        # 计算传递函数 H(fx, fy)
        with torch.no_grad():
            # 角谱传递函数
            kz = np.sqrt(self.k**2 - (2*np.pi*FX)**2 - (2*np.pi*FY)**2)
            # 处理衰减波（evanescent waves）
            kz = np.where(np.iscomplex(kz) | (kz.imag != 0), kz.real, kz)
            kz = np.where(kz**2 < 0, 0, kz)
            
            H_filter = np.exp(1j * kz * self.distance)
            H_filter = torch.tensor(H_filter, dtype=torch.complex64, device=input_field.device)
        
        # 角谱法传播
        output_field = torch.zeros_like(input_field, dtype=torch.complex64)
        
        for i in range(batch_size):
            # 傅里叶变换
            U_fft = torch.fft.fft2(input_field[i])
            # 应用传递函数
            U_prop_fft = U_fft * H_filter
            # 逆傅里叶变换
            U_prop = torch.fft.ifft2(U_prop_fft)
            output_field[i] = U_prop
            
        return output_field

# ==================== 2. 衍射层实现 ====================
class DiffractiveLayer(nn.Module):
    """
    单个衍射层：相位调制 + 衍射传播
    """
    def __init__(self, feature_size=200, wavelength=0.75e-3, 
                 pixel_size=0.04, distance=40e-3):
        super().__init__()
        self.feature_size = feature_size
        
        # 可学习的相位调制参数
        self.phase_modulation = nn.Parameter(
            torch.randn(feature_size, feature_size) * 0.1
        )
        
        # 衍射传播模块
        self.propagation = DiffractivePropagation(
            wavelength=wavelength,
            pixel_size=pixel_size,
            distance=distance
        )
        
    def forward(self, input_field):
        """
        前向传播函数：实现衍射层的相位调制和传播
        
        参数:
            input_field: 输入复数光场，形状为 [batch, height, width]
                - batch: 批次大小
                - height, width: 光场的空间维度
        
        返回:
            output_field: 调制并传播后的复数光场，形状为 [batch, height, width]
        
        物理原理:
            1. 相位调制: 通过可学习的相位参数对输入光场进行调制
            2. 衍射传播: 将调制后的光场传播到下一层平面
        """
        # 获取输入光场的批次大小和空间维度
        batch_size, H, W = input_field.shape
        
        # 1. 生成相位掩码
        # 利用复数指数函数创建相位调制掩码: exp(i*phi)
        # self.phase_modulation 是可学习的相位参数
        phase_mask = torch.exp(1j * self.phase_modulation)
        
        # 扩展相位掩码以匹配批次大小
        # unsqueeze(0) 添加批次维度，expand 扩展到整个批次
        phase_mask = phase_mask.unsqueeze(0).expand(batch_size, -1, -1)
        
        # 2. 应用相位调制
        # 复数乘法实现相位调制: U' = U * exp(i*phi)
        modulated_field = input_field * phase_mask
        
        # 3. 执行衍射传播
        # 通过 DiffractivePropagation 模块计算光场传播
        output_field = self.propagation(modulated_field)
        
        # 返回传播后的光场
        return output_field

# ==================== 3. 5层D²NN网络 ====================
class D2NN_5Layer(nn.Module):
    """
    5层衍射神经网络，用于MNIST手写数字分类
    架构: 输入 → 衍射层1 → 衍射层2 → 衍射层3 → 衍射层4 → 衍射层5 → 输出
    """
    def __init__(self, input_size=28, feature_size=200, num_classes=10):
        super().__init__()
        
        # 超参数（基于论文）
        self.wavelength = 0.75e-3  # 0.75mm，太赫兹波段
        self.pixel_size = 0.04    # 40μm像素尺寸
        self.layer_distance = 40e-3  # 层间距40mm
        
        # 输入编码：将MNIST图像编码为光场振幅
        self.input_encoder = nn.Sequential(
            nn.Linear(input_size*input_size, feature_size*feature_size),
            nn.ReLU()
        )
        
        # 5个衍射层
        self.diffractive_layers = nn.ModuleList([
            DiffractiveLayer(
                feature_size=feature_size,
                wavelength=self.wavelength,
                pixel_size=self.pixel_size,
                distance=self.layer_distance
            ) for _ in range(5)
        ])
        
        # 输出检测器：10个区域对应10个数字
        self.output_detectors = nn.Parameter(
            torch.randn(num_classes, feature_size, feature_size) * 0.1
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(num_classes, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        
    def encode_input(self, x):
        """将输入图像编码为光场振幅分布"""
        batch_size = x.shape[0]
        # 展平并编码
        x_flat = x.view(batch_size, -1)
        encoded = self.input_encoder(x_flat)
        
        # 重塑为2D振幅分布，相位初始为0
        amplitude = encoded.view(batch_size, self.diffractive_layers[0].feature_size, 
                                self.diffractive_layers[0].feature_size)
        phase = torch.zeros_like(amplitude)
        
        # 创建复数光场: U = A * exp(i*phi)
        input_field = amplitude * torch.exp(1j * phase)
        
        return input_field
    
    def forward(self, x):
        """前向传播"""
        # 1. 输入编码
        input_field = self.encode_input(x)
        
        # 2. 通过5个衍射层
        current_field = input_field
        for layer in self.diffractive_layers:
            current_field = layer(current_field)
        
        # 3. 计算输出平面光强
        output_intensity = torch.abs(current_field)**2
        
        # 4. 与检测器模板匹配（模拟物理检测）
        batch_size = output_intensity.shape[0]
        detector_scores = []
        
        for i in range(batch_size):
            scores = []
            for class_idx in range(10):
                # 计算该检测器区域的总光强
                detector_mask = torch.sigmoid(self.output_detectors[class_idx])
                region_intensity = torch.sum(output_intensity[i] * detector_mask)
                scores.append(region_intensity)
            
            detector_scores.append(torch.stack(scores))
        
        detector_output = torch.stack(detector_scores)
        
        # 5. 分类决策
        final_output = self.classifier(detector_output.view(batch_size, -1))
        
        return final_output, output_intensity, detector_output

# ==================== 4. 训练与评估 ====================
def train_d2nn():
    """训练5层D²NN"""
    # 超参数
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 5  # 减少epoch数以加速演示
    feature_size = 80  # 减小尺寸以加速训练
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 数据加载
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 初始化模型
    model = D2NN_5Layer(input_size=28, feature_size=feature_size, num_classes=10)
    model = model.to(device)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 训练循环
    train_losses = []
    train_accuracies = []
    
    print("开始训练5层D²NN...")
    print("="*50)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            
            # 前向传播
            output, _, _ = model(data)
            loss = criterion(output, target)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 统计
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch+1}/{num_epochs} | '
                      f'Batch: {batch_idx}/{len(train_loader)} | '
                      f'Loss: {loss.item():.4f}')
        
        # 计算epoch统计
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)
        
        print(f'Epoch {epoch+1} 完成: '
              f'平均损失: {epoch_loss:.4f}, '
              f'准确率: {epoch_acc:.2f}%')
        print("-"*50)
        
        # 测试集评估
        test_acc = evaluate_model(model, test_loader, device)
        print(f'测试集准确率: {test_acc:.2f}%')
        print("="*50)
    
    return model, train_losses, train_accuracies

def evaluate_model(model, test_loader, device):
    """评估模型性能"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output, _, _ = model(data)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    accuracy = 100. * correct / total
    return accuracy

# ==================== 5. 可视化工具 ====================
def visualize_d2nn_components(model, sample_data):
    """可视化D²NN关键组件"""
    device = next(model.parameters()).device
    
    with torch.no_grad():
        # 获取各层输出
        input_field = model.encode_input(sample_data.unsqueeze(0).to(device))
        
        fields = [input_field]
        current_field = input_field
        
        for layer in model.diffractive_layers:
            # 获取相位调制
            phase_mask = torch.exp(1j * layer.phase_modulation)
            
            # 应用调制
            modulated = current_field * phase_mask.unsqueeze(0)
            
            # 传播
            current_field = layer.propagation(modulated)
            fields.append(current_field)
    
    # 创建可视化
    fig, axes = plt.subplots(2, 6, figsize=(18, 6))
    fig.suptitle('5层D²NN工作原理可视化', fontsize=16, fontweight='bold')
    
    titles = ['输入光场', '层1调制后', '层1输出', 
              '层2调制后', '层2输出', '层3调制后',
              '层3输出', '层4调制后', '层4输出',
              '层5调制后', '层5输出', '最终光强']
    
    for idx, ax in enumerate(axes.flat):
        if idx < len(fields):
            if idx == len(fields) - 1:
                # 最终光强
                intensity = torch.abs(fields[-1][0]**2)
                im = ax.imshow(intensity.cpu().numpy(), cmap='hot')
                ax.set_title(titles[idx])
            else:
                # 振幅或相位
                if idx % 2 == 0:  # 调制前/输出
                    data = torch.angle(fields[idx//2][0]).cpu().numpy()
                    cmap = 'hsv'
                else:  # 调制后
                    data = torch.abs(fields[idx//2][0]).cpu().numpy()
                    cmap = 'gray'
                
                im = ax.imshow(data, cmap=cmap)
                ax.set_title(titles[idx])
        
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

def plot_training_history(train_losses, train_accuracies):
    """绘制训练历史"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # 损失曲线
    ax1.plot(train_losses, 'b-', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('训练损失曲线')
    ax1.grid(True, alpha=0.3)
    
    # 准确率曲线
    ax2.plot(train_accuracies, 'r-', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('训练准确率曲线')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# ==================== 6. 物理仿真验证 ====================
def simulate_physical_propagation():
    """模拟真实物理衍射传播"""
    # 参数设置（基于论文实验）
    wavelength = 0.75e-3  # 0.75mm，太赫兹
    pixel_size = 0.04e-3  # 40μm
    N = 200  # 网格尺寸
    z = 40e-3  # 传播距离40mm
    
    # 创建输入场（模拟数字"5"）
    x = np.linspace(-N/2, N/2, N) * pixel_size
    y = np.linspace(-N/2, N/2, N) * pixel_size
    X, Y = np.meshgrid(x, y)
    
    # 圆形孔径中的数字图案
    radius = N * pixel_size / 4
    circle_mask = (X**2 + Y**2) < radius**2
    
    # 简单数字图案
    digit_pattern = np.zeros((N, N))
    digit_pattern[80:120, 80:120] = 1  # 中心方块
    
    input_field = digit_pattern * circle_mask
    
    # 角谱法传播
    k = 2 * np.pi / wavelength
    
    # 空间频率
    dfx = 1 / (N * pixel_size)
    dfy = 1 / (N * pixel_size)
    fx = np.fft.fftfreq(N, d=pixel_size)
    fy = np.fft.fftfreq(N, d=pixel_size)
    FX, FY = np.meshgrid(fx, fy)
    
    # 传递函数
    kz = np.sqrt(k**2 - (2*np.pi*FX)**2 - (2*np.pi*FY)**2 + 0j)
    H = np.exp(1j * kz * z)
    
    # 传播计算
    U_in_fft = np.fft.fft2(input_field)
    U_out_fft = U_in_fft * H
    U_out = np.fft.ifft2(U_out_fft)
    
    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    axes[0].imshow(input_field, cmap='gray')
    axes[0].set_title('输入场 (数字"5")')
    axes[0].axis('off')
    
    axes[1].imshow(np.angle(H), cmap='hsv')
    axes[1].set_title('传递函数相位')
    axes[1].axis('off')
    
    axes[2].imshow(np.abs(U_out)**2, cmap='hot')
    axes[2].set_title('传播后光强')
    axes[2].axis('off')
    
    plt.suptitle('物理衍射传播仿真', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    return input_field, U_out

# ==================== 7. 主程序 ====================
def main():
    """主函数：完整演示D²NN"""
    print("="*60)
    print("衍射神经网络（D²NN）完整演示系统")
    print("="*60)
    
    # 1. 物理原理演示
    print("\n1. 物理衍射传播仿真...")
    input_field, output_field = simulate_physical_propagation()
    
    # 2. 训练D²NN模型
    print("\n2. 训练5层D²NN模型...")
    model, train_losses, train_accuracies = train_d2nn()
    
    # 3. 可视化训练过程
    print("\n3. 可视化训练历史...")
    plot_training_history(train_losses, train_accuracies)
    
    # 4. 可视化网络组件
    print("\n4. 可视化D²NN各层状态...")
    # 获取一个测试样本
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    sample_data, _ = test_dataset[0]
    
    visualize_d2nn_components(model, sample_data)
    
    # 5. 性能分析
    print("\n5. 模型性能分析...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_loader = DataLoader(
        datasets.MNIST('./data', train=False, transform=transform),
        batch_size=64, shuffle=False
    )
    
    final_accuracy = evaluate_model(model, test_loader, device)
    print(f"最终测试准确率: {final_accuracy:.2f}%")
    
    # 6. 与论文结果对比
    print("\n6. 与原始论文结果对比:")
    print("   - 论文报告 (5层D²NN, MNIST): 91.75% (数值), 88% (实验)")
    print(f"   - 本实现 (5层D²NN, MNIST): {final_accuracy:.2f}% (模拟)")
    print("   注: 本实现为简化版，实际性能受网络规模、训练时长影响")
    
    # 7. 保存模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'train_losses': train_losses,
        'train_accuracies': train_accuracies
    }, 'd2nn_5layer_model.pth')
    
    print("\n模型已保存为 'd2nn_5layer_model.pth'")
    print("="*60)
    print("演示完成！")

# ==================== 8. 理论公式总结 ====================
def print_theoretical_summary():
    """打印理论公式总结"""
    print("\n" + "="*60)
    print("D²NN 核心理论公式总结")
    print("="*60)
    
    formulas = [
        ("1. 角谱传播", r"$E_z(k_x, k_y) = E_0(k_x, k_y) \cdot \exp\left(iz\sqrt{k^2 - k_x^2 - k_y^2}\right)$"),
        ("2. 传播算子", r"$\mathcal{P}_z\{E(x,y)\} = \mathcal{F}^{-1}\left\{\mathcal{F}\{E(x,y)\} \cdot H(k_x,k_y)\right\}$"),
        ("3. 调制函数", r"$T_l^{(m,n)} = a_l^{(m,n)} \cdot \exp\left(i\phi_l^{(m,n)}\right) \cdot \delta(x-m\Delta, y-n\Delta)$"),
        ("4. 层间传播", r"$U_{l+1} = \mathcal{P}_{d_l}\{U_l \cdot T_l\}$"),
        ("5. 输出检测", r"$I_{\text{out}} = |U_L|^2, \quad \hat{y} = \arg\max_j \sum_{\text{region}_j} I_{\text{out}}$"),
        ("6. 损失函数", r"$\mathcal{L} = -\sum_{j=1}^{10} y_j \log\left(\frac{\exp(s_j)}{\sum_k \exp(s_k)}\right)$"),
        ("7. 反向传播", r"$\frac{\partial \mathcal{L}}{\partial \phi_l} = \text{Re}\left\{\frac{\partial \mathcal{L}}{\partial U_{l+1}} \cdot \frac{\partial U_{l+1}}{\partial \phi_l}\right\}$")
    ]
    
    for name, formula in formulas:
        print(f"\n{name}:")
        print(f"  {formula}")
    
    print("\n" + "="*60)

# 运行主程序
if __name__ == "__main__":
    # 打印理论总结
    print_theoretical_summary()
    
    # 运行完整演示
    print("\n运行完整演示...")
    main()