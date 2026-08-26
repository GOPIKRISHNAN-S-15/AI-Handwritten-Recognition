import numpy as np
from PIL import Image
import streamlit as st
import sys
import __main__

try:
    import torch
    import torch.nn as nn
    from huggingface_hub import hf_hub_download
    TORCH_CTC_AVAILABLE = True
except ImportError:
    TORCH_CTC_AVAILABLE = False
    torch = None
    nn = None
    hf_hub_download = None

class CharacterMapper:
    def __init__(self):
        chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,;:!?\'"()-')
        self.chars = sorted(list(chars))
        self.char2idx = {c: i+1 for i, c in enumerate(self.chars)}
        self.idx2char = {i+1: c for i, c in enumerate(self.chars)}
        self.idx2char[0] = ''  # CTC blank
        self.num_classes = len(self.chars) + 1

    def encode(self, text):
        return [self.char2idx[c] for c in text if c in self.char2idx]

    def decode(self, indices):
        chars, prev = [], None
        for idx in indices:
            if idx != 0 and idx != prev and idx in self.idx2char:
                chars.append(self.idx2char[idx])
            prev = idx
        return ''.join(chars)

# Patch main to allow checkpoint loading if it expects CharacterMapper
__main__.CharacterMapper = CharacterMapper

if TORCH_CTC_AVAILABLE:
    class CRNN(nn.Module):
        """CNN-BiLSTM-CTC for Handwriting Recognition"""
        def __init__(self, img_height=128, num_chars=80, hidden_size=256, num_layers=2):
            super(CRNN, self).__init__()
            self.cnn = nn.Sequential(
                nn.Conv2d(1, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2),
                nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2),
                nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
                nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d((2, 1)),
                nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
                nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(), nn.MaxPool2d((2, 1)),
                nn.Conv2d(512, 512, 2), nn.BatchNorm2d(512), nn.ReLU(),
            )
            self.map2seq = nn.Linear(512 * 7, hidden_size)
            self.rnn = nn.LSTM(hidden_size, hidden_size, num_layers, bidirectional=True,
                               dropout=0.3 if num_layers > 1 else 0, batch_first=True)
            self.fc = nn.Linear(hidden_size * 2, num_chars + 1)

        def forward(self, x):
            conv = self.cnn(x)
            b, c, h, w = conv.size()
            conv = conv.permute(0, 3, 1, 2).reshape(b, w, c * h)
            seq = self.map2seq(conv)
            rnn_out, _ = self.rnn(seq)
            output = self.fc(rnn_out)
            return torch.nn.functional.log_softmax(output, dim=2)
else:
    class CRNN:
        pass

@st.cache_resource(show_spinner="Loading CTC Baseline Model...")
def load_ctc_model():
    if not TORCH_CTC_AVAILABLE:
        raise RuntimeError("PyTorch or huggingface_hub is not installed in the environment.")
    ckpt_path = hf_hub_download(repo_id="ismatsamadov/handwriting-recognition-iam", filename="best_model.pth")
    
    # The training script pickled CharacterMapper under __main__
    # When running under Streamlit, sys.modules['__main__'] is the streamlit CLI.
    # We must inject CharacterMapper into sys.modules['__main__'] so the unpickler can find it.
    import sys
    sys.modules['__main__'].CharacterMapper = CharacterMapper
    
    try:
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except Exception as e:
        import traceback
        st.error(f"CTC Load Traceback:\n```\n{traceback.format_exc()}\n```")
        raise e
        
    char_mapper = checkpoint['char_mapper']
    model = CRNN(num_chars=len(char_mapper.chars))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, char_mapper

def run_ctc_on_lines(line_images, target_height=128, target_width=512):
    model, char_mapper = load_ctc_model()
    results = []
    
    with torch.no_grad():
        for img_arr in line_images:
            if isinstance(img_arr, np.ndarray):
                img = Image.fromarray(img_arr).convert('L')
            else:
                img = img_arr.convert('L')
                
            w, h = img.size
            # Resize proportionally to target_height=128
            new_w = int(target_height * (w / h))
            img = img.resize((new_w, target_height), Image.LANCZOS)
            
            img = np.array(img, dtype=np.float32) / 255.0
            img = (img - 0.5) / 0.5
            img_tensor = torch.FloatTensor(img).unsqueeze(0).unsqueeze(0)
            
            output = model(img_tensor)
            pred_indices = output.argmax(dim=2).squeeze(0).tolist()
            text = char_mapper.decode(pred_indices)
            results.append(text)
            
    return results
