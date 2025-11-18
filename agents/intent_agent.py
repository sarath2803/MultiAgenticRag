# agents/intent_agent.py
import os
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
import pickle

MODEL_DIR = os.path.join("models", "intent_model")
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_INTENTS = ["fact", "analysis", "summary", "visual"]

class IntentClassifier(nn.Module):
    def __init__(self, input_dim=384, hidden_dim=128, num_classes=4):
        super(IntentClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.model(x)

class IntentAgent:
    def __init__(self, model_dir=MODEL_DIR):
        self.model_dir = model_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedder = SentenceTransformer(EMBED_MODEL_NAME, device=self.device)

        self.intent_labels = DEFAULT_INTENTS
        self.label_to_idx = {lbl: i for i, lbl in enumerate(self.intent_labels)}

        model_path = os.path.join(model_dir, "intent_classifier.pt")

        if os.path.exists(model_path):
            self.classifier = IntentClassifier(num_classes=len(self.intent_labels))
            self.classifier.load_state_dict(torch.load(model_path, map_location=self.device))
            self.classifier.to(self.device)
            self.classifier.eval()
        else:
            print("⚠️ Warning: Trained intent classifier not found. Using rule-based fallback.")
            self.classifier = None

    def predict(self, text: str) -> str:
        text = text.strip()
        if not text:
            return "fact"

        # Rule-based fallback
        if self.classifier is None:
            print("🔍 Using rule-based classifier")
            return self._rule_based(text)
        print("🔍 Using deep learning model")
        # Encode and classify
        emb = self.embedder.encode(text, convert_to_tensor=True).to(self.device)
        with torch.no_grad():
            logits = self.classifier(emb)
            pred_idx = torch.argmax(logits, dim=-1).item()
            return self.intent_labels[pred_idx]

    def _rule_based(self, t: str) -> str:
        t = t.lower()
        if any(w in t for w in ["figure", "chart", "diagram", "image", "plot", "table", "graph"]):
            return "visual"
        if any(w in t for w in ["summarize", "summary", "overview", "key points"]):
            return "summary"
        if any(w in t for w in ["compare", "contrast", "analysis", "synthesize", "why", "evaluate"]):
            return "analysis"
        return "fact"

    @staticmethod
    def save_trained_model(model, out_dir=MODEL_DIR):
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "intent_classifier.pt")
        torch.save(model.state_dict(), path)
        print(f"✅ Saved deep learning intent model to {path}")


