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
        self.text_encoder = CLIPTextModel.from_pretrained(repo, subfolder="text_encoder")
        self.text_encoder.to(device)
        self.devid = device
    def encode(self, text: str) -> torch.Tensor:
        """
        parses the text into chunks for the tokenizer
        then encodes it with the text encoder, adds it into one tensor
        """
        send_to_tkizer = []
        #check if the text is larger than the tokenizer's max length
        if len(text) > self.tokenizer.model_max_length:
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
            tokenized.append(self.tokenizer.encode(part, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt"))
        #send each part through the text encoder
        weights = []
        for part in tokenized:
            #move to cuda IF CUDA is available
            weights.append(self.text_encoder(part.to(self.devid))[0])
        stacked = torch.stack(weights)
        total = stacked.sum(dim=0)
        return total
