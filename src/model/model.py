"""
Model sieci neuronowej dla rekomendacji filmów (PyTorch).

Architektura:
- Wielowarstwowa sieć Feed-Forward (MLP)
- Dropout dla regularyzacji
- Przewidywanie oceny użytkownika (regression)
"""

import torch
import torch.nn as nn


class MovieRecommenderNet(nn.Module):
    """
    Neural network dla rekomendacji filmów.
    
    Prosta architektura MLP z dropout.
    Input: feature vector (numerical + encoded categorical)
    Output: predicted rating (0-5)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [256, 128, 64],
        dropout_rate: float = 0.3
    ):
        """
        Args:
            input_dim: Liczba input features
            hidden_dims: Lista wymiarów hidden layers
            dropout_rate: Dropout rate dla regularyzacji
        """
        super(MovieRecommenderNet, self).__init__()
        
        self.input_dim = input_dim
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Output layer (regression - 1 neuron)
        layers.append(nn.Linear(prev_dim, 1))
        # Sigmoid * 5 żeby wymusić zakres 0-5
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)
        self.output_scale = 5.0  # Skala outputu do zakresu 0-5
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch_size, input_dim)
            
        Returns:
            Predicted ratings (batch_size, 1) w zakresie 0-5
        """
        return self.model(x) * self.output_scale


class MovieRecommenderNetDeep(nn.Module):
    """
    Głębsza wersja modelu z residual connections.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [512, 256, 128, 64],
        dropout_rate: float = 0.4
    ):
        """
        Args:
            input_dim: Liczba input features
            hidden_dims: Lista wymiarów hidden layers
            dropout_rate: Dropout rate
        """
        super(MovieRecommenderNetDeep, self).__init__()
        
        self.input_dim = input_dim
        
        # Input projection
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.Dropout(dropout_rate)
        )
        
        # Hidden layers z residual connections
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.hidden_layers.append(
                ResidualBlock(hidden_dims[i], hidden_dims[i + 1], dropout_rate)
            )
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dims[-1], 1),
            nn.Sigmoid()
        )
        self.output_scale = 5.0
    
    def forward(self, x):
        """Forward pass."""
        x = self.input_layer(x)
        
        for layer in self.hidden_layers:
            x = layer(x)
        
        return self.output_layer(x) * self.output_scale


class MovieRecommenderNetAdvanced(nn.Module):
    """
    Zaawansowana architektura z:
    - Głębszymi warstwami
    - Residual connections
    - Attention mechanism na features
    - Layer Normalization
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [512, 384, 256, 128, 64],
        dropout_rate: float = 0.4,
        use_attention: bool = True
    ):
        super(MovieRecommenderNetAdvanced, self).__init__()
        
        self.input_dim = input_dim
        self.use_attention = use_attention
        
        # Optional attention na input features
        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(input_dim, input_dim // 4),
                nn.ReLU(),
                nn.Linear(input_dim // 4, input_dim),
                nn.Sigmoid()
            )
        
        # Input projection
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Hidden layers z residual connections
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.hidden_layers.append(
                ResidualBlock(hidden_dims[i], hidden_dims[i + 1], dropout_rate, use_layer_norm=True)
            )
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dims[-1], 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        self.output_scale = 5.0
    
    def forward(self, x):
        """Forward pass z attention."""
        # Attention na input features
        if self.use_attention:
            attention_weights = self.attention(x)
            x = x * attention_weights
        
        x = self.input_layer(x)
        
        for layer in self.hidden_layers:
            x = layer(x)
        
        return self.output_layer(x) * self.output_scale


class ResidualBlock(nn.Module):
    """Residual block z skip connection."""
    
    def __init__(self, input_dim: int, output_dim: int, dropout_rate: float = 0.3, use_layer_norm: bool = False):
        super(ResidualBlock, self).__init__()
        
        norm_layer = nn.LayerNorm(output_dim) if use_layer_norm else nn.BatchNorm1d(output_dim)
        
        self.main_path = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            norm_layer,
            nn.Dropout(dropout_rate)
        )
        
        # Skip connection (jeśli wymiary się nie zgadzają)
        if input_dim != output_dim:
            self.skip_connection = nn.Linear(input_dim, output_dim)
        else:
            self.skip_connection = nn.Identity()
    
    def forward(self, x):
        """Forward pass z residual connection."""
        return self.main_path(x) + self.skip_connection(x)


def create_model(input_dim: int, model_type: str = 'standard', **kwargs):
    """
    Factory function do tworzenia modeli.
    
    Args:
        input_dim: Liczba input features
        model_type: 'standard', 'deep', lub 'advanced'
        **kwargs: Dodatkowe argumenty dla modelu
        
    Returns:
        Model PyTorch
        
    Architektury:
        - standard: Prosta MLP (256->128->64) - szybka, ~86k params
        - deep: Głębsza MLP z residual connections (512->256->128->64) - ~340k params
        - advanced: Najlepsza - attention + residual + LayerNorm (512->384->256->128->64) - ~530k params
    """
    if model_type == 'standard':
        return MovieRecommenderNet(input_dim, **kwargs)
    elif model_type == 'deep':
        return MovieRecommenderNetDeep(input_dim, **kwargs)
    elif model_type == 'advanced':
        return MovieRecommenderNetAdvanced(input_dim, **kwargs)
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Choose: 'standard', 'deep', 'advanced'")


if __name__ == "__main__":
    # Test modelu
    print("🧪 Test modelu...\n")
    
    # Parametry
    input_dim = 177  # Z prepare_training_data.py
    batch_size = 32
    
    # Stwórz model
    model_standard = create_model(input_dim, model_type='standard')
    model_deep = create_model(input_dim, model_type='deep')
    
    print(f"Standard Model:")
    print(f"  Parameters: {sum(p.numel() for p in model_standard.parameters()):,}")
    print(f"  Trainable: {sum(p.numel() for p in model_standard.parameters() if p.requires_grad):,}")
    
    print(f"\nDeep Model:")
    print(f"  Parameters: {sum(p.numel() for p in model_deep.parameters()):,}")
    print(f"  Trainable: {sum(p.numel() for p in model_deep.parameters() if p.requires_grad):,}")
    
    # Test forward pass
    dummy_input = torch.randn(batch_size, input_dim)
    
    with torch.no_grad():
        output_standard = model_standard(dummy_input)
        output_deep = model_deep(dummy_input)
    
    print(f"\n✅ Forward pass test:")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Standard output shape: {output_standard.shape}")
    print(f"  Deep output shape: {output_deep.shape}")
    print(f"  Standard output range: [{output_standard.min():.2f}, {output_standard.max():.2f}]")
    print(f"  Deep output range: [{output_deep.min():.2f}, {output_deep.max():.2f}]")
    
    print("\n🎉 Model działa poprawnie!")

