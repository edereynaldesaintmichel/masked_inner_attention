import json
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

import torch.optim as optim
from itertools import islice
import torch.nn as nn
from torch.nn import functional as F

# Hyperparams
BATCH_SIZE = 1000
LEARNING_RATE = 1e-3
NUM_EPOCHS = 1000
N_HEAD = 64
N_LAYER = 1
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_GRAD_NORM = 1e7 # No need for gradient clipping thanks to the LayerNorm at the end of each MultiHead Block
MAX_DROPOUT_PROB = 0.3

torch.manual_seed(42)


class EloisHead(nn.Module):
    def __init__(self, n_embd, head_size, dropout = 0.1):
        super().__init__()
        self.head_size = head_size
        self.values_proj = nn.Linear(n_embd, head_size, bias=False)
        self.cov = nn.Parameter(torch.ones(head_size, head_size)*0.1)
        self.loadings = nn.Parameter(torch.ones(head_size, head_size)*0.1)
        # nn.init.xavier_normal_(self.loadings)
        # nn.init.xavier_normal_(self.cov)

    def forward(self, x):
        values = self.values_proj(x[:,0,:])  # (B, hs)
        mask = self.values_proj(x[:,1,:]) # (B, hs)
        one_minus_mask = self.values_proj(1-x[:,1,:])
        weighted_avg_coefficients = torch.eye(self.head_size).to(DEVICE) + torch.diag_embed(mask) @ self.cov @ torch.diag_embed(one_minus_mask)# (B, hs, hs)
        
        # weighted_avg_coefficients = weighted_avg_coefficients / weighted_avg_coefficients.sum(dim = -1, keepdim=True)
        weighted_avg_coefficients = F.softmax(weighted_avg_coefficients, dim=-1)
        loadings = self.loadings * weighted_avg_coefficients # check if this is a real pointwise operation
        # weighted_avg_coefficients = self.dropout(weighted_avg_coefficients)
        
        # Add dimension to values using unsqueeze
        values = values.unsqueeze(1)  # (B, 1, hs)
        out = values @ loadings.transpose(-2,-1) # (B, 1, hs)
        out = out.squeeze(1)  # (B, hs) - Remove the middle dimension
        # out = self.batch_norm(out)  # Now BatchNorm1d will work
        
        return out
    

class MultiHeadLayer(nn.Module):
    """Multiple EloisHead in parallel."""

    def __init__(self, n_embd, n_head, head_size, dropout = 0.1):
        super().__init__()
        self.heads = nn.ModuleList([EloisHead(n_embd=n_embd, head_size=head_size, dropout=dropout) for _ in range(n_head)])
        self.proj = nn.Linear(head_size * n_head, n_embd)
        self.dropout = nn.Dropout(dropout)
        self.layerNorm = nn.LayerNorm(n_embd)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        out = self.layerNorm(out)
        # out = torch.cat([out, x[:,1,:]], dim=1)

        return out


class FeedFoward(nn.Module):
    """a simple linear layer followed by a non-linearity"""

    def __init__(self, n_embd, dropout = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            # nn.Linear(n_embd, 1*n_embd),
            nn.GELU(),
            nn.Linear(1*n_embd, n_embd),
            # nn.Dropout(dropout),
        )

    def forward(self, x):
        out = self.net(x)
        # out = torch.cat([out, x[:,1,:]], dim=1)

        return out
    
class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.mh = MultiHeadLayer(n_embd=n_embd, n_head=n_head, head_size=head_size)
        self.ffwd = FeedFoward(n_embd)

    def forward(self, x):
        out = self.mh(x)
        # out = out + x[:,0,:] 
        out = self.ffwd(out)
        out = torch.cat([out.unsqueeze(1), x[:,1,:].unsqueeze(1)], dim=1)

        return out


class FinancialDropout(nn.Module):
    def __init__(self, drop_prob):
        super().__init__()
        self.drop_prob = drop_prob
        
    def forward(self, x, dropout_factor):
        if not self.training:
            return x
            
        # x shape is (batch_size, 2, n_embd)
        values = x[:, 0, :]  # (batch_size, n_embd)
        masks = x[:, 1, :]   # (batch_size, n_embd)
        
        # Create random dropout mask
        random_mask = (torch.rand_like(values) > self.drop_prob * dropout_factor).float()
        
        # Generate random scaling factors between 0.5 and 2 for each sample in the batch
        scale_factors = torch.rand(values.shape[0], 1, device=values.device) * 1.5 + 0.5  # generates numbers between 0.5 and 2
        
        # Apply dropout and scaling to values, only dropout to masks
        new_values = values * random_mask * scale_factors
        new_masks = masks * random_mask
        
        # Recombine into original format
        return torch.stack([new_values, new_masks], dim=1)

class EloisNet(nn.Module):
    def __init__(self, n_embd, n_head, n_layer, output_size, dropout_prob=0.5):
        super().__init__()
        self.financial_dropout = FinancialDropout(dropout_prob)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head=n_head) for _ in range(n_layer)]
        )
        self.lm_head = nn.Sequential(
            # nn.Linear(n_embd, 1*n_embd),
            nn.GELU(),
            nn.Linear(1*n_embd, output_size)
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0, std=0.02)

    def forward(self, input, targets=None, dropout_factor=1):
        input = self.financial_dropout(input, dropout_factor)  # Apply financial dropout during training
        block_output = self.blocks(input)  # (B,2,n_emb)
        logits = self.lm_head(block_output[:,0,:])  # (B, output_size)

        if targets is None:
            return logits, None
            
        loss = F.l1_loss(logits, targets) #+ torch.norm((block_output[:,0,:] - input[:,0,:]) * input[:,1,:])
        return logits, loss

def load_data(filepath='financial_statements.json', output_vector_fields=['netincomeloss']):
    # Load financial statements
    with open(filepath, 'r+') as file:
        financial_statements = json.load(file)

    # Count field occurrences
    all_fields_counter = {}
    # financial_statements = dict(islice(financial_statements.items(), 5))
    for financial_statement in financial_statements.values():
        for train_yearly_report in financial_statement:
            for key, value in train_yearly_report.items():
                if not isinstance(value, (int, float)):
                    continue
                all_fields_counter[key] = all_fields_counter.get(key, 0) + 1

    # Get top 128 most common fields
    sorted_all_fields_counter = dict(
        sorted(all_fields_counter.items(), 
              key=lambda item: item[1], 
              reverse=True)[:256]
    )

    # sorted_all_fields_counter = {"netincomeloss": all_fields_counter["netincomeloss"]} # Just for testing purpose

    test_input_data = []
    test_ground_truth = []

    
    train_input_data = []
    train_ground_truth = []
    
    for financial_statement in financial_statements.values():
        financial_statement.reverse()
        if len(financial_statement) < 5:
            continue
            
        train_input_vector = []
        train_masking_vector = []

        train_output_vector = []

        test_input_vector = []
        test_masking_vector = []

        test_output_vector = []
        
        # Get input data from first 3 years
        for index, train_yearly_report in enumerate(financial_statement):
            if (index > 3):
                break
            for key in sorted_all_fields_counter:
                value = train_yearly_report.get(key, 0)
                try:
                    value = float(value)/1e9
                except:
                    value = 0
                masking_value = 0 if key in train_yearly_report else 1
                if index > 0:
                    test_input_vector.append(value)
                    test_masking_vector.append(masking_value)
                if index < 3:
                    train_input_vector.append(value)
                    train_masking_vector.append(masking_value)
        

        train_yearly_report = financial_statement[3]
        test_yearly_report = financial_statement[4]

        for key in output_vector_fields:
            train_value = train_yearly_report.get(key, None)
            test_value = test_yearly_report.get(key, None)
            if train_value is not None:
                train_value /= 1e9
            if test_value is not None:
                test_value /= 1e9
            train_output_vector.append(train_value)
            test_output_vector.append(test_value)
        
        train_input_data.append((train_input_vector, train_masking_vector))
        train_ground_truth.append(train_output_vector)

        test_input_data.append((test_input_vector, test_masking_vector))
        test_ground_truth.append(test_output_vector)

    return train_input_data, train_ground_truth, test_input_data, test_ground_truth, len(train_input_data[0][0])

def prepare_sub_data(input_data, ground_truth):
    """
    Helper function to filter out samples with None values in ground truth
    and convert the data to tensors.
    """
    # Filter out any samples with None values in ground truth
    valid_samples = []
    valid_targets = []
    
    for idx, (sample, target) in enumerate(zip(input_data, ground_truth)):
        if None in target:
            continue
        valid_samples.append(sample)
        valid_targets.append(target)
    
    # Convert to tensors
    values = torch.FloatTensor([x[0] for x in valid_samples])
    mask = torch.FloatTensor([x[1] for x in valid_samples])
    targets = torch.FloatTensor(valid_targets)
    
    # Create proper input format (B, 2, n_embd)
    data = torch.stack([values, mask], dim=1)
    
    return data, targets

def prepare_data(train_input_data, train_ground_truth, test_input_data, test_ground_truth):
    """
    Main function to prepare training and evaluation data.
    Evaluation data combines both validation and test data.
    """
    # Prepare training data
    X_train, y_train = prepare_sub_data(train_input_data, train_ground_truth)
    
    # Prepare evaluation data (combine validation and test data)
    X_eval, y_eval = prepare_sub_data(test_input_data, test_ground_truth)
    
    return X_train, y_train, X_eval, y_eval

def create_dataloaders(X_train, y_train, X_val, y_val, batch_size):
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    return train_loader, val_loader

def train_epoch(model, train_loader, optimizer, device, index):
    model.train()
    total_loss = 0
    length = 0
    
    for batch_idx, (data, targets) in enumerate(train_loader):
        data, targets = data.to(device), targets.to(device)
        
        optimizer.zero_grad()
        logits, loss = model(data, targets, (index**0.2)/(NUM_EPOCHS**0.2))
        
        if loss is None:
            continue
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()
        
        total_loss += loss.item()
        length += 1
        # Add gradient monitoring
        # for name, param in model.named_parameters():
        #     if param.grad is not None:
        #         grad_norm = param.grad.norm()
        #         if grad_norm > 10:
        #             print(f"Large gradient in {name}: {grad_norm}")
    
    return total_loss / length

def validate(model, val_loader, device):
    model.eval()
    total_loss = 0
    length = 0
    
    with torch.no_grad():
        for data, targets in val_loader:
            data, targets = data.to(device), targets.to(device)
            logits, loss = model(data, targets)
            
            if loss is None:
                continue
                
            total_loss += loss.item()
            length += 1
    
    return total_loss / length

def main():
        # Fields to predict
    output_vector_fields = ['netincomeloss']
    OUTPUT_SIZE = len(output_vector_fields)
    # Load and prepare data
    train_input_data, train_ground_truth, test_input_data, test_ground_truth, n_embd = load_data()
    X_train, y_train, X_val, y_val = prepare_data(train_input_data, train_ground_truth, test_input_data, test_ground_truth,)
    train_loader, val_loader = create_dataloaders(X_train, y_train, X_val, y_val, BATCH_SIZE)
    
    # Initialize model
    model = EloisNet(
        n_embd=n_embd,
        n_head=N_HEAD,
        n_layer=N_LAYER,
        output_size=OUTPUT_SIZE,
        dropout_prob=MAX_DROPOUT_PROB,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")
    
    # Initialize optimizer
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(NUM_EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, DEVICE, epoch)
        val_loss = validate(model, val_loader, DEVICE)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'test_model.pt')
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

if __name__ == '__main__':
    main()

