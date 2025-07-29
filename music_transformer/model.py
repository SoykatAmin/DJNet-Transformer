"""
Core transformer model for music transition generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model, max_length=5000):
        super().__init__()
        
        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism"""
    
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear transformations and reshape
        Q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention computation
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply mask if provided
        if mask is not None:
            # Ensure mask has correct dimensions [batch, n_heads, seq_len, seq_len]
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model)
        
        return self.w_o(context)


class FeedForward(nn.Module):
    """Position-wise feed-forward network"""
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerEncoderLayer(nn.Module):
    """Single transformer encoder layer"""
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_out = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-forward with residual connection
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))
        
        return x


class TransformerEncoder(nn.Module):
    """Stack of transformer encoder layers"""
    
    def __init__(self, d_model, n_heads, n_layers, d_ff, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x


class TransformerDecoderLayer(nn.Module):
    """Single transformer decoder layer with cross-attention"""
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn_forward = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn_backward = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, forward_context, backward_context, tgt_mask=None, src_mask=None):
        # Self-attention on target sequence
        self_attn_out = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_out))
        
        # Cross-attention with forward context
        cross_attn_forward = self.cross_attn_forward(x, forward_context, forward_context, src_mask)
        x = self.norm2(x + self.dropout(cross_attn_forward))
        
        # Cross-attention with backward context
        cross_attn_backward = self.cross_attn_backward(x, backward_context, backward_context, src_mask)
        x = self.norm3(x + self.dropout(cross_attn_backward))
        
        # Feed-forward
        ff_out = self.feed_forward(x)
        x = self.norm4(x + self.dropout(ff_out))
        
        return x


class TransformerDecoder(nn.Module):
    """Stack of transformer decoder layers"""
    
    def __init__(self, d_model, n_heads, n_layers, d_ff, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
    def forward(self, x, forward_context, backward_context, tgt_mask=None, src_mask=None):
        for layer in self.layers:
            x = layer(x, forward_context, backward_context, tgt_mask, src_mask)
        return x


class MusicTransitionTransformer(nn.Module):
    """Main transformer model for music transition generation"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Input/Output projections
        self.input_projection = nn.Linear(config.n_mels, config.d_model)
        self.output_projection = nn.Linear(config.d_model, config.n_mels)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(config.d_model, config.max_sequence_length)
        
        # Dual encoders for forward and backward context
        self.forward_encoder = TransformerEncoder(
            config.d_model, config.n_heads, config.n_layers, 
            config.d_ff, config.dropout
        )
        self.backward_encoder = TransformerEncoder(
            config.d_model, config.n_heads, config.n_layers, 
            config.d_ff, config.dropout
        )
        
        # Decoder for transition generation
        self.decoder = TransformerDecoder(
            config.d_model, config.n_heads, config.n_layers, 
            config.d_ff, config.dropout
        )
        
        # Dropout and layer norm
        self.dropout = nn.Dropout(config.dropout)
        self.layer_norm = nn.LayerNorm(config.d_model)
        
        # Initialize weights
        self.init_weights()
        
    def init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
    
    def create_causal_mask(self, size):
        """Create causal mask for decoder self-attention"""
        mask = torch.tril(torch.ones(size, size))
        return mask  # Shape: [size, size]
    
    def forward(self, forward_context, backward_context, target=None, teacher_forcing=True):
        """
        Forward pass of the model
        
        Args:
            forward_context: Spectrogram of preceding segment [batch, time, mels]
            backward_context: Spectrogram of following segment [batch, time, mels]
            target: Target transition sequence for training [batch, time, mels]
            teacher_forcing: Whether to use teacher forcing during training
        """
        batch_size = forward_context.size(0)
        device = forward_context.device
        
        # Debug: Check input shapes and fix if needed
        if forward_context.dim() == 2:
            # If input is [batch, time*mels], reshape to [batch, time, mels]
            time_steps = forward_context.size(1) // self.config.n_mels
            forward_context = forward_context.view(batch_size, time_steps, self.config.n_mels)
            backward_context = backward_context.view(batch_size, time_steps, self.config.n_mels)
            if target is not None:
                target = target.view(target.size(0), target.size(1) // self.config.n_mels, self.config.n_mels)
        
        # Project input spectrograms to model dimension
        forward_emb = self.input_projection(forward_context)  # [batch, time, d_model]
        backward_emb = self.input_projection(backward_context)
        
        # Add positional encoding
        forward_emb = self.pos_encoding(forward_emb)
        backward_emb = self.pos_encoding(backward_emb)
        
        # Apply dropout
        forward_emb = self.dropout(forward_emb)
        backward_emb = self.dropout(backward_emb)
        
        # Encode contexts
        forward_encoded = self.forward_encoder(forward_emb)
        backward_encoded = self.backward_encoder(backward_emb)
        
        if self.training and target is not None and teacher_forcing:
            # Training mode with teacher forcing
            target_emb = self.input_projection(target)
            target_emb = self.pos_encoding(target_emb)
            target_emb = self.dropout(target_emb)
            
            # Create causal mask for decoder
            tgt_len = target.size(1)
            causal_mask = self.create_causal_mask(tgt_len).to(device)
            
            # Decode transition
            decoded = self.decoder(
                target_emb, forward_encoded, backward_encoded,
                tgt_mask=causal_mask
            )
            
            # Project back to spectrogram space
            output = self.output_projection(decoded)
            return output
            
        else:
            # Inference mode - generate transition autoregressively
            return self.generate_transition(forward_encoded, backward_encoded)
    
    def generate_transition(self, forward_encoded, backward_encoded, temperature=1.0):
        """
        Generate transition sequence autoregressively
        
        Args:
            forward_encoded: Encoded forward context
            backward_encoded: Encoded backward context
            temperature: Sampling temperature for generation
        """
        batch_size = forward_encoded.size(0)
        device = forward_encoded.device
        
        # Initialize with silence (zeros)
        generated = torch.zeros(batch_size, 1, self.config.n_mels, device=device)
        
        for step in range(self.config.transition_length):
            # Project current sequence to model dimension
            target_emb = self.input_projection(generated)
            target_emb = self.pos_encoding(target_emb)
            
            # Create causal mask
            tgt_len = generated.size(1)
            causal_mask = self.create_causal_mask(tgt_len).to(device)
            
            # Decode next step
            decoded = self.decoder(
                target_emb, forward_encoded, backward_encoded,
                tgt_mask=causal_mask
            )
            
            # Project to spectrogram space and get last time step
            next_logits = self.output_projection(decoded[:, -1:, :])  # [batch, 1, mels]
            
            # Apply temperature sampling (for spectrograms, we use different sampling)
            if temperature > 0:
                next_probs = torch.sigmoid(next_logits / temperature)
                # For spectrograms, we can sample from the continuous distribution
                next_step = next_probs + torch.randn_like(next_probs) * 0.1 * temperature
                next_step = torch.clamp(next_step, 0, 1)  # Keep in valid range
            else:
                next_step = torch.sigmoid(next_logits)
            
            # Append to generated sequence
            generated = torch.cat([generated, next_step], dim=1)
        
        return generated[:, 1:, :]  # Remove initial silence token
