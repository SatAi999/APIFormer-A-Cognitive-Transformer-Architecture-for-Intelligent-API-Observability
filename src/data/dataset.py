import torch
from torch.utils.data import Dataset
import random
from typing import List, Dict, Any, Tuple
import numpy as np

class APIFormerDataset(Dataset):
    """PyTorch Dataset preparing API traffic sequences for Self-Supervised and Multi-Task learning.

    
    Includes tasks:
    - Masked API Modeling (MAM)
    - Next API Prediction
    - Latency Forecasting
    - Response Code Prediction
    - Contrastive Session Learning (dual augmented views)
    """
    
    def __init__(self, 
                 encoded_sessions: List[Dict[str, List[Any]]], 
                 vocab: Dict[str, int], 
                 mask_prob: float = 0.15,
                 max_len: int = 128):
        self.sessions = encoded_sessions
        self.vocab = vocab
        self.mask_prob = mask_prob
        self.max_len = max_len
        
        self.pad_idx = vocab.get("[PAD]", 0)
        self.mask_idx = vocab.get("[MASK]", 2)
        
    def __len__(self) -> int:
        return len(self.sessions)
        
    def _pad_sequence(self, seq: List[int], pad_val: int) -> torch.Tensor:
        padded = seq[:self.max_len]
        padded += [pad_val] * (self.max_len - len(padded))
        return torch.tensor(padded, dtype=torch.long)
        
    def _pad_float_sequence(self, seq: List[float], pad_val: float) -> torch.Tensor:
        padded = seq[:self.max_len]
        padded += [pad_val] * (self.max_len - len(padded))
        return torch.tensor(padded, dtype=torch.float)

    def _apply_masking(self, endpoints: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Applies MLM (15% masking rate) to the endpoint sequence."""
        labels = []
        masked_inputs = []
        
        for token in endpoints[:self.max_len]:
            prob = random.random()
            if prob < self.mask_prob:
                prob /= self.mask_prob
                # 80% replace with [MASK]
                if prob < 0.8:
                    masked_inputs.append(self.mask_idx)
                # 10% replace with random token
                elif prob < 0.9:
                    # Choose a random index from vocabulary that is not special
                    masked_inputs.append(random.randint(5, len(self.vocab) - 1))
                # 10% keep unchanged
                else:
                    masked_inputs.append(token)
                labels.append(token)
            else:
                masked_inputs.append(token)
                labels.append(-100) # Ignore loss index in PyTorch CrossEntropy
                
        # Padding
        padded_inputs = masked_inputs + [self.pad_idx] * (self.max_len - len(masked_inputs))
        padded_labels = labels + [-100] * (self.max_len - len(labels))
        
        return torch.tensor(padded_inputs, dtype=torch.long), torch.tensor(padded_labels, dtype=torch.long)

    def _augment_session(self, session: Dict[str, Any]) -> Dict[str, Any]:
        """Applies data augmentation (timing jitter, minor dropout) for contrastive learning views."""
        aug = {}
        for cat, vals in session.items():
            if isinstance(vals, list):
                aug[cat] = list(vals)
            else:
                aug[cat] = vals
        
        # 1. Timing Jitter
        aug["latency"] = [max(0.0, float(x) + np.random.normal(0, 5.0)) for x in aug["latency"]]
        aug["time_gap"] = [max(0.0, float(x) + np.random.normal(0, 0.2)) for x in aug["time_gap"]]
        
        # 2. Token Dropout (replace a token with UNK with 5% prob)
        unk_idx = self.vocab.get("[UNK]", 1)
        for i in range(len(aug["endpoint"])):
            if random.random() < 0.05:
                aug["endpoint"][i] = unk_idx
                
        return aug

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sess = self.sessions[idx]
        
        # Original features
        ep_seq = sess["endpoint"]
        method_seq = sess["method"]
        status_seq = sess["status"]
        service_seq = sess["service"]
        device_seq = sess["device"]
        location_seq = sess["location"]
        env_seq = sess["env"]
        auth_seq = sess["auth"]
        
        latency_seq = sess["latency"]
        payload_seq = sess["payload"]
        time_gap_seq = sess["time_gap"]
        
        # Pad categoricals
        padded_ep = self._pad_sequence(ep_seq, self.pad_idx)
        padded_method = self._pad_sequence(method_seq, self.pad_idx)
        padded_status = self._pad_sequence(status_seq, self.pad_idx)
        padded_service = self._pad_sequence(service_seq, self.pad_idx)
        padded_device = self._pad_sequence(device_seq, self.pad_idx)
        padded_location = self._pad_sequence(location_seq, self.pad_idx)
        padded_env = self._pad_sequence(env_seq, self.pad_idx)
        padded_auth = self._pad_sequence(auth_seq, self.pad_idx)
        
        # Pad numericals
        padded_latency = self._pad_float_sequence(latency_seq, 0.0)
        padded_payload = self._pad_float_sequence(payload_seq, 0.0)
        padded_time_gap = self._pad_float_sequence(time_gap_seq, 0.0)
        
        # 1. Masked API Modeling
        masked_ep, mam_labels = self._apply_masking(ep_seq)
        
        # 2. Next API Prediction (causal shifting)
        next_ep_target = self._pad_sequence(ep_seq[1:] + [self.pad_idx], self.pad_idx)
        
        # 3. Contrastive Session Views
        view1 = self._augment_session(sess)
        view2 = self._augment_session(sess)
        
        padded_v1_ep = self._pad_sequence(view1["endpoint"], self.pad_idx)
        padded_v2_ep = self._pad_sequence(view2["endpoint"], self.pad_idx)
        padded_v1_lat = self._pad_float_sequence(view1["latency"], 0.0)
        padded_v2_lat = self._pad_float_sequence(view2["latency"], 0.0)
        
        # Determine sequence mask (for padding masking inside transformer)
        seq_len = min(len(ep_seq), self.max_len)
        padding_mask = torch.zeros(self.max_len, dtype=torch.bool)
        padding_mask[seq_len:] = True # True indicates positions to be masked out
        
        # Downstream fine-tuning targets
        anomaly_seq = sess.get("anomaly_labels", [0] * len(ep_seq))
        padded_anomaly = self._pad_sequence(anomaly_seq, 0)
        intent_label = int(sess.get("intent_label", 0))
        bot_label = int(sess.get("bot_label", 0))
        
        return {
            # Core inputs
            "endpoint": padded_ep,
            "method": padded_method,
            "status": padded_status,
            "service": padded_service,
            "device": padded_device,
            "location": padded_location,
            "env": padded_env,
            "auth": padded_auth,
            "latency": padded_latency,
            "payload": padded_payload,
            "time_gap": padded_time_gap,
            "padding_mask": padding_mask,
            
            # MLM (Masked API Modeling) targets
            "masked_endpoint": masked_ep,
            "mam_labels": mam_labels,
            
            # Next prediction targets
            "next_endpoint_targets": next_ep_target,
            
            # Contrastive views
            "v1_endpoint": padded_v1_ep,
            "v2_endpoint": padded_v2_ep,
            "v1_latency": padded_v1_lat,
            "v2_latency": padded_v2_lat,
            
            # Downstream targets
            "anomaly_labels": padded_anomaly,
            "intent_label": torch.tensor(intent_label, dtype=torch.long),
            "bot_label": torch.tensor(bot_label, dtype=torch.long)
        }
