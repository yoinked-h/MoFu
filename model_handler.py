"""
Model handling for MoFU
"""
from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextModelWithProjection
import torch

class TokenizerModels:
    """
    Wrapper for tokenizer and text encoder models
    """
    def __init__(self, repo: str = "Birchlabs/wd-1-5-beta3-unofficial", device: str = "cuda", sdxl: bool = False):
        if not isinstance(repo, str):
            if sdxl:
                repo = "cagliostrolab/animagine-xl-3.1"
            else:
                repo = "Birchlabs/wd-1-5-beta3-unofficial"
        self.sdxl = sdxl
        if not sdxl:
            self.tokenizer = CLIPTokenizer.from_pretrained(repo, subfolder="tokenizer")
            self.text_encoder = CLIPTextModel.from_pretrained(repo, subfolder="text_encoder")
        else:
            self.tokenizer = CLIPTokenizer.from_pretrained(repo, subfolder="tokenizer")
            self.text_encoder = CLIPTextModelWithProjection.from_pretrained(repo, subfolder="text_encoder")
            self.text_encoder_2 = CLIPTextModel.from_pretrained(repo, subfolder="text_encoder_2")
            self.text_encoder_2.to(device)
            
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
            if not self.sdxl:
                weights.append(self.text_encoder(part.to(self.devid))[0])
            else:
                a = self.text_encoder(part.to(self.devid))[0]
                b = self.text_encoder_2(part.to(self.devid))[0]
                weights.append([a, b])
        if not self.sdxl:
            stacked = torch.stack(weights)
            total = stacked.sum(dim=0)
        else:
            reordered = [[],[]]
            for part in weights:
                reordered[0].append(part[0])
                reordered[1].append(part[1])
            stacked = [None, None]
            stacked[0] = torch.stack(reordered[0])
            stacked[1] = torch.stack(reordered[1])
            total = [stacked[0].sum(dim=0), stacked[1].sum(dim=0)]
        return total
