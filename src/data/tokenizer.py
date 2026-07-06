import re
import json
import os
from typing import List, Dict, Any, Tuple
from src.data.synthetic_gen import UnifiedAPIEvent
from src.utils.logger import logger

class APITokenizer:
    """Custom tokenizer that cleans URL paths and builds vocabularies for all categorical API dimensions."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get("tokenizer", {})
        self.mask_token = self.config.get("mask_token", "[MASK]")
        self.pad_token = self.config.get("pad_token", "[PAD]")
        self.unk_token = self.config.get("unk_token", "[UNK]")
        self.cls_token = self.config.get("cls_token", "[CLS]")
        self.sep_token = self.config.get("sep_token", "[SEP]")
        
        # Regexes to detect and extract path variables
        self.variable_regexes = [
            (re.compile(r'\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b'), "{uuid}"),
            (re.compile(r'\b\d+\b'), "{id}")
        ]
        
        # Vocabularies
        self.vocabs = {
            "endpoint": {},
            "method": {},
            "status": {},
            "service": {},
            "device": {},
            "location": {},
            "env": {},
            "auth": {}
        }
        
        self.inverse_vocabs = {}
        
    def normalize_endpoint(self, endpoint: str) -> str:
        """Replaces dynamic segments like IDs and UUIDs in URLs with placeholders."""
        normalized = endpoint
        for regex, placeholder in self.variable_regexes:
            normalized = regex.sub(placeholder, normalized)
        return normalized

    def fit(self, sessions: Dict[str, List[UnifiedAPIEvent]]) -> None:
        """Scans sessions to populate vocabulary mappings for all attributes."""
        logger.info("Fitting tokenizer on sessions...")
        
        temp_sets = {key: set() for key in self.vocabs.keys()}
        
        for events in sessions.values():
            for e in events:
                temp_sets["endpoint"].add(self.normalize_endpoint(e.endpoint))
                temp_sets["method"].add(e.http_method.upper())
                temp_sets["status"].add(str(e.status_code))
                temp_sets["service"].add(e.service_name)
                temp_sets["device"].add(e.device)
                temp_sets["location"].add(e.geo_location)
                temp_sets["env"].add(e.environment)
                temp_sets["auth"].add(e.authentication)
                
        # Initialize vocab maps with special tokens
        special_tokens = [self.pad_token, self.unk_token, self.mask_token, self.cls_token, self.sep_token]
        
        for category, unique_tokens in temp_sets.items():
            vocab = {tok: idx for idx, tok in enumerate(special_tokens)}
            idx = len(special_tokens)
            
            for tok in sorted(list(unique_tokens)):
                if tok not in vocab:
                    vocab[tok] = idx
                    idx += 1
            self.vocabs[category] = vocab
            
        self._build_inverse_vocabs()
        logger.info(f"Tokenizer fitted. Endpoint vocab size: {len(self.vocabs['endpoint'])}, Service vocab size: {len(self.vocabs['service'])}")

    def _build_inverse_vocabs(self) -> None:
        self.inverse_vocabs = {
            cat: {v: k for k, v in vocab.items()}
            for cat, vocab in self.vocabs.items()
        }

    def encode_event(self, e: UnifiedAPIEvent) -> Dict[str, int]:
        """Maps a single event's categorical dimensions to token integer IDs."""
        norm_ep = self.normalize_endpoint(e.endpoint)
        
        return {
            "endpoint": self.vocabs["endpoint"].get(norm_ep, self.vocabs["endpoint"][self.unk_token]),
            "method": self.vocabs["method"].get(e.http_method.upper(), self.vocabs["method"][self.unk_token]),
            "status": self.vocabs["status"].get(str(e.status_code), self.vocabs["status"][self.unk_token]),
            "service": self.vocabs["service"].get(e.service_name, self.vocabs["service"][self.unk_token]),
            "device": self.vocabs["device"].get(e.device, self.vocabs["device"][self.unk_token]),
            "location": self.vocabs["location"].get(e.geo_location, self.vocabs["location"][self.unk_token]),
            "env": self.vocabs["env"].get(e.environment, self.vocabs["env"][self.unk_token]),
            "auth": self.vocabs["auth"].get(e.authentication, self.vocabs["auth"][self.unk_token])
        }

    def encode_session(self, session: List[UnifiedAPIEvent]) -> Dict[str, List[Any]]:
        """Converts a sequence of events into vectors of token IDs along with numerical features."""
        encoded = {cat: [] for cat in self.vocabs.keys()}
        encoded["latency"] = []
        encoded["payload"] = []
        encoded["time_gap"] = []
        
        for e in session:
            evt_tokens = self.encode_event(e)
            for cat, tid in evt_tokens.items():
                encoded[cat].append(tid)
            # Logarithmic transformation for highly skewed numerical values to ease gradient flow
            encoded["latency"].append(float(e.latency_ms))
            encoded["payload"].append(float(e.payload_size_bytes))
            encoded["time_gap"].append(float(e.time_since_previous_request))
            
        # Extract fine-tuning downstream labels
        intent_map = {"normal": 0, "503_cascade": 1, "credential_stuffing": 2, "bot_scraping": 3, "sequence_abuse": 4}
        event_types = [e.event_type for e in session]
        
        encoded["anomaly_labels"] = [0 if et == "normal" else 1 for et in event_types]
        
        # Session-level labels (based on majority or initial event)
        sess_intent = event_types[0] if len(event_types) > 0 else "normal"
        encoded["intent_label"] = intent_map.get(sess_intent, 0)
        
        is_bot = 1 if (len(session) > 0 and session[0].device == "bot-agent") else 0
        encoded["bot_label"] = is_bot
        
        return encoded

    def save(self, file_path: str) -> None:
        """Saves vocab maps to JSON file."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(self.vocabs, f, indent=2)
        logger.info(f"Saved tokenizer vocabs to: {file_path}")
        
    def load(self, file_path: str) -> None:
        """Loads vocab maps from JSON file."""
        with open(file_path, "r") as f:
            self.vocabs = json.load(f)
        self._build_inverse_vocabs()
        logger.info(f"Loaded tokenizer vocabs from: {file_path}")
