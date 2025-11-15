# agents/vision_agent.py
import os
from typing import Dict, Any
import torch
from PIL import Image



# Use BLIP base captioning (small) and fallback to filename caption
try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
    BLIP_AVAILABLE = True
except Exception:
    BLIP_AVAILABLE = False

class VisionAgent:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = None
        self.model = None
        if BLIP_AVAILABLE:
            try:
                # small model (Salesforce/blip-image-captioning-base)
                self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
                self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
                self.model.to(self.device)
            except Exception as e:
                print("BLIP load failed, falling back to filename captions.", e)
                self.processor = None
                self.model = None

    def describe_image(self, img_path: str, max_length: int = 64) -> Dict[str, Any]:
        
        if self.processor and self.model:
            image = Image.open(img_path).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=max_length)
                caption = self.processor.decode(outputs[0], skip_special_tokens=True)
            return {"caption": caption}
        else:
            fn = os.path.basename(img_path)
            caption = f"Image file named {fn}. Possibly a chart or figure extracted from the document."
            return {"caption": caption}
