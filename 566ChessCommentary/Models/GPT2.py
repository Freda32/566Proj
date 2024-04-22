import torch
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer
from Utils import get_chess_tokens, dataset_tokens

GPT2_TYPE = "gpt2"

class GPT2:
    def __init__(self):
        print("Initialization\n")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.tokenizer = GPT2Tokenizer.from_pretrained(GPT2_TYPE)
        special_tokens_dict = {
            'pad_token': '[PAD]',
            'additional_special_tokens': dataset_tokens
        }
        self.tokenizer.add_special_tokens(special_tokens_dict)
        self.tokenizer.add_tokens(get_chess_tokens())

        self.configuration = GPT2Config.from_pretrained(GPT2_TYPE)
        self.model = GPT2LMHeadModel(self.configuration)
        
        # Resize token embeddings to accommodate new tokens
        self.model.resize_token_embeddings(len(self.tokenizer))

        self.model.to(self.device)

    def load_model(self, model_path):
        print("Loading model\n")
        state_dict = torch.load(model_path, map_location=self.device)

        # Ensure the model is correctly sized before loading the state dict
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
