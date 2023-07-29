"""
Model handling for MoFU
"""
from transformers import CLIPTextModel, CLIPTokenizer
import torch

class TokenizerModels:
    """
    Wrapper for tokenizer and text encoder models
    """
    def __init__(self, repo: str = "Birchlabs/wd-1-5-beta3-unofficial", device: str = "cuda"):
        if not isinstance(repo, str):
            repo = "Birchlabs/wd-1-5-beta3-unofficial"
        self.tokenizer = CLIPTokenizer.from_pretrained(repo, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(repo, subfolder="text_encoder", device=device)
    def encode(self, text: str) -> torch.Tensor:
        """
        parses the text into chunks for the tokenizer, then encodes it with the text encoder
        """
        send_to_tkizer = []
        #check if the text is larger than the tokenizer's max length
        if len(text) > self.tokenizer.model_max_length:
            #split text on every "," then join as many times as long as it is under the tokenizer's max length
            textparts = text.split(",")
            blob = ""
            for part in textparts:
                if len(blob) + part < self.tokenizer.model_max_length:
                    blob += part
                else:
                    send_to_tkizer.append(blob)
                    blob = ""
        else:
            send_to_tkizer = [text]
        tokenized = []
        for part in send_to_tkizer:
            tokenized.append(self.tokenizer.encode(part))
        #send each part through the text encoder
        

            


