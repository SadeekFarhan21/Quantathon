import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer models"""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [seq_len, batch_size, embedding_dim]
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerModel(nn.Module):
    """Transformer model for sequence classification with improved regularization"""
    def __init__(self, input_size, d_model=128, nhead=4, num_layers=2, 
                 dim_feedforward=512, dropout=0.3, num_classes=3):
        super(TransformerModel, self).__init__()
        
        self.model_type = 'Transformer'
        # Input embedding
        self.src_embedding = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer encoder with higher dropout
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout + 0.1,  # Increase dropout to reduce overfitting
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # Output layer with extra regularization
        self.fc1 = nn.Linear(d_model, dim_feedforward)
        self.bn1 = nn.BatchNorm1d(dim_feedforward)  # Add batch normalization
        self.dropout1 = nn.Dropout(dropout + 0.1)
        self.fc2 = nn.Linear(dim_feedforward, num_classes)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Use smaller standard deviation for initialization
            module.weight.data.normal_(mean=0.0, std=0.01)  
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        
    def forward(self, src):
        # Input shape: [batch_size, seq_len, input_size]
        
        # Check for NaN values and replace them
        if torch.isnan(src).any():
            src = torch.nan_to_num(src, nan=0.0)
            
        src = self.src_embedding(src)  # [batch_size, seq_len, d_model]
        src = self.pos_encoder(src)
        
        # Create a dummy mask (no masking)
        src_mask = None
        
        # Transformer processing
        output = self.transformer_encoder(src, src_mask)  # [batch_size, seq_len, d_model]
        
        # Take the representation from the last sequence element
        output = output[:, -1, :]  # [batch_size, d_model]
        
        # Classification head with batch normalization
        output = self.fc1(output)
        output = self.bn1(output)  # Apply batch normalization
        output = torch.relu(output)
        output = self.dropout1(output)
        output = self.fc2(output)
        
        return output
