import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Model, GPT2Config
from tiktoken import get_encoding

class GPT2CLMMLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Initialize the base GPT2 model
        self.gpt2 = GPT2Model(config)
        
        # MLM head
        self.mlm_head = nn.Linear(config.n_embd, config.vocab_size)
        
        # CLM head (reuse GPT2's language modeling head)
        self.clm_head = nn.Linear(config.n_embd, config.vocab_size)
        
        # Initialize weights
        self.init_weights()
        
    def init_weights(self):
        # Initialize MLM head
        nn.init.normal_(self.mlm_head.weight, std=0.02)
        nn.init.zeros_(self.mlm_head.bias)
        
        # Initialize CLM head
        nn.init.normal_(self.clm_head.weight, std=0.02)
        nn.init.zeros_(self.clm_head.bias)
    
    def forward(self, input_ids, attention_mask=None, labels=None, mlm_labels=None):
        # Get GPT2 base model outputs
        outputs = self.gpt2(input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        
        # CLM loss (85%)
        clm_logits = self.clm_head(hidden_states)
        clm_loss = None
        if labels is not None:
            # Shift logits and labels for causal LM
            shift_logits = clm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            clm_loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), 
                                      shift_labels.view(-1))
        
        # MLM loss (15%)
        mlm_logits = self.mlm_head(hidden_states)
        mlm_loss = None
        if mlm_labels is not None:
            mlm_loss = F.cross_entropy(mlm_logits.view(-1, mlm_logits.size(-1)),
                                      mlm_labels.view(-1),
                                      ignore_index=-100)
        
        # Combine losses with 85% CLM and 15% MLM weights
        total_loss = None
        if clm_loss is not None and mlm_loss is not None:
            total_loss = 0.85 * clm_loss + 0.15 * mlm_loss
        
        return {
            'loss': total_loss,
            'clm_loss': clm_loss,
            'mlm_loss': mlm_loss,
            'clm_logits': clm_logits,
            'mlm_logits': mlm_logits
        }

# Helper function to create the model
def create_gpt2_clm_mlm(pretrained_model_name='gpt2'):
    # Load GPT2 configuration
    config = GPT2Config.from_pretrained(pretrained_model_name)
    
    # Create custom model
    model = GPT2CLMMLM(config)
    
    # Load pre-trained GPT2 weights into the base model
    pretrained_gpt2 = GPT2Model.from_pretrained(pretrained_model_name)
    model.gpt2.load_state_dict(pretrained_gpt2.state_dict())
    
    return model

# Initialize tokenizer
def get_tiktoken_tokenizer(encoding_name='gpt2'):
    return get_encoding(encoding_name)