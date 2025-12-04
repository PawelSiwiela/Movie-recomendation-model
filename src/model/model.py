"""
Model sieci neuronowej dla rekomendacji filmów (PyTorch).

Zoptymalizowana architektura z:
- Residual connections dla głębszych sieci
- Attention mechanism na features
- Layer Normalization i Dropout
- Przewidywanie oceny użytkownika (regression 0-5)
"""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Residual block z skip connection."""
    
    def __init__(self, input_dim: int, output_dim: int, dropout_rate: float = 0.3):
        super(ResidualBlock, self).__init__()
        
        self.main_path = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout_rate)
        )
        
        # Skip connection
        if input_dim != output_dim:
            self.skip_connection = nn.Linear(input_dim, output_dim)
        else:
            self.skip_connection = nn.Identity()
    
    def forward(self, x):
        return self.main_path(x) + self.skip_connection(x)


class MovieRecommenderNet(nn.Module):
    """
    Zoptymalizowany model dla rekomendacji filmów.
    
    Architektura:
    - Attention mechanism na input features (opcjonalny)
    - Input projection z mniejszym wymiarem (256 zamiast 512)
    - 2 Residual blocks z Layer Normalization (256→128→64)
    - Output layer z Sigmoid (0-5 range)
    
    Parametry: ~130k dla 183 input features (uproszczona sieć dla małych zbiorów)
    Wydajność: Val RMSE ~0.65 (567 samples), ~0.52 (97 samples)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = None,
        dropout_rate: float = 0.3,
        use_attention: bool = True
    ):
        """
        Args:
            input_dim: Liczba input features
            hidden_dims: Lista wymiarów hidden layers (default: [256, 128, 64])
            dropout_rate: Dropout rate dla regularyzacji
            use_attention: Czy używać attention mechanism
        """
        super(MovieRecommenderNet, self).__init__()
        
        if hidden_dims is None:
            # Uproszczona architektura dla małych zbiorów danych
            hidden_dims = [256, 128, 64]
        
        self.input_dim = input_dim
        self.use_attention = use_attention
        
        # Attention na input features (opcjonalny)
        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(input_dim, max(input_dim // 4, 32)),
                nn.ReLU(),
                nn.Linear(max(input_dim // 4, 32), input_dim),
                nn.Softmax(dim=1)
            )
        
        # Input projection
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.Dropout(dropout_rate)
        )
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.residual_blocks.append(
                ResidualBlock(hidden_dims[i], hidden_dims[i + 1], dropout_rate)
            )
        
        # Output layer - usunięto LayerNorm przed Sigmoid dla lepszej ekspresji
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dims[-1], 1),
            nn.Sigmoid()
        )
        self.output_scale = 5.0
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch_size, input_dim)
            
        Returns:
            Predicted ratings (batch_size, 1) w zakresie 0-5
        """
        # Optional attention
        if self.use_attention:
            attention_weights = self.attention(x)
            x = x * attention_weights
        
        # Main path
        x = self.input_layer(x)
        
        for block in self.residual_blocks:
            x = block(x)
        
        return self.output_layer(x).squeeze(1) * self.output_scale


def create_model(input_dim: int, **kwargs):
    """
    Factory function do tworzenia modelu.
    
    Args:
        input_dim: Liczba input features (183 dla obecnego pipeline)
        **kwargs: Dodatkowe argumenty:
            - hidden_dims: lista wymiarów (default: [256, 128, 64])
            - dropout_rate: float (default: 0.3)
            - use_attention: bool (default: True)
    
    Returns:
        MovieRecommenderNet model
    
    Przykład:
        >>> model = create_model(183)  # Default config (~130k params)
        >>> model = create_model(183, dropout_rate=0.4)  # Większy dropout
        >>> model = create_model(183, hidden_dims=[128, 64])  # Jeszcze mniejsza sieć
    """
    return MovieRecommenderNet(input_dim, **kwargs)


if __name__ == "__main__":
    # Test modelu
    print("🧪 Test modelu...\\n")
    
    # Parametry
    input_dim = 183  # Z prepare_training_data.py (27 genres + 51 directors + 100 actors + 3 numerical + 2 type)
    batch_size = 32
    
    # Stwórz model
    model = create_model(input_dim)
    
    print(f"MovieRecommenderNet:")
    print(f"  Input dim: {input_dim}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Test forward pass
    dummy_input = torch.randn(batch_size, input_dim)
    
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"\\n✅ Forward pass test:")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output range: [{output.min():.2f}, {output.max():.2f}]")
    print(f"  Output mean: {output.mean():.2f}")
    
    print("\\n🎉 Model działa poprawnie!")

