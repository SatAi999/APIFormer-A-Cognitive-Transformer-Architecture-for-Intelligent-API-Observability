import pandas as pd
import json
from typing import Dict, Any, Optional, List
from src.utils.logger import logger

class DatasetInspector:
    """Automatically inspects arbitrary tabular or JSON datasets to discover the schema, 

    columns, data types, and mapping relationships for API traffic logs."""
    
    def __init__(self):
        # Target fields we wish to detect
        self.target_fields = {
            "timestamp": ["timestamp", "time", "ts", "@timestamp", "date"],
            "session_id": ["session_id", "sessionid", "sess_id", "session"],
            "trace_id": ["trace_id", "traceid", "trace", "span_id", "span"],
            "correlation_id": ["correlation_id", "correlationid", "corr_id", "corrid"],
            "user_id": ["user_id", "userid", "user", "uid", "username"],
            "service_name": ["service_name", "servicename", "service", "app", "application"],
            "endpoint": ["endpoint", "uri", "url", "path", "request_path"],
            "http_method": ["http_method", "method", "verb", "request_method"],
            "status_code": ["status_code", "status", "code", "response_code"],
            "latency": ["latency", "latency_ms", "duration", "time_taken", "response_time"],
            "payload_size": ["payload_size", "payload_size_bytes", "bytes", "response_size", "size"]
        }

    def _jaro_winkler_sim(self, s1: str, s2: str) -> float:
        """Computes Jaro-Winkler similarity between two strings."""
        s1, s2 = s1.lower().strip(), s2.lower().strip()
        if s1 == s2:
            return 1.0
        
        len1, len2 = len(s1), len(s2)
        if len1 == 0 or len2 == 0:
            return 0.0
        
        match_bound = max(len1, len2) // 2 - 1
        s1_matches = [False] * len1
        s2_matches = [False] * len2
        
        matches = 0
        transpositions = 0
        
        for i in range(len1):
            start = max(0, i - match_bound)
            end = min(i + match_bound + 1, len2)
            for j in range(start, end):
                if not s2_matches[j] and s1[i] == s2[j]:
                    s1_matches[i] = True
                    s2_matches[j] = True
                    matches += 1
                    break
                    
        if matches == 0:
            return 0.0
            
        k = 0
        for i in range(len1):
            if s1_matches[i]:
                while not s2_matches[k]:
                    k += 1
                if s1[i] != s2[k]:
                    transpositions += 1
                k += 1
                
        transpositions //= 2
        
        jaro = (matches / len1 + matches / len2 + (matches - transpositions) / matches) / 3.0
        
        # Winkler modification
        prefix_len = 0
        for i in range(min(4, min(len1, len2))):
            if s1[i] == s2[i]:
                prefix_len += 1
            else:
                break
                
        return jaro + prefix_len * 0.1 * (1.0 - jaro)

    def _guess_field_by_values(self, col_name: str, sample_series: pd.Series) -> Optional[str]:
        """Examines column data content to guess if it matches standard fields."""
        non_nulls = sample_series.dropna().head(20)
        if non_nulls.empty:
            return None
            
        # Try to infer by content
        # HTTP Method
        if all(isinstance(val, str) and val.upper() in ["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"] for val in non_nulls):
            return "http_method"
            
        # Status code (int between 100 and 599)
        if sample_series.dtype in ['int64', 'float64']:
            if all(100 <= val < 600 for val in non_nulls):
                return "status_code"
                
        # Timestamp check
        if sample_series.dtype == 'object':
            try:
                pd.to_datetime(non_nulls, errors='raise')
                return "timestamp"
            except (ValueError, TypeError):
                pass
                
        # Endpoint check (strings starting with /)
        if all(isinstance(val, str) and val.startswith('/') for val in non_nulls):
            return "endpoint"
            
        return None

    def inspect(self, file_path: str) -> Dict[str, Any]:
        """Inspects file and returns a configuration mapping table for normalizer."""
        logger.info(f"Inspecting file: {file_path}")
        
        # Determine format
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path, nrows=500)
        elif file_path.endswith('.json'):
            try:
                df = pd.read_json(file_path, lines=True, nrows=500)
            except Exception:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                if isinstance(data, list):
                    df = pd.DataFrame(data[:500])
                else:
                    df = pd.DataFrame([data])
        elif file_path.endswith('.parquet'):
            df = pd.read_parquet(file_path)[:500]
        else:
            raise ValueError("Unsupported file format. Use CSV, JSON or Parquet.")
            
        detected_mapping = {}
        missing_report = {}
        
        for col in df.columns:
            # Check missing values
            null_count = df[col].isnull().sum()
            missing_report[col] = float(null_count / len(df))
            
            # 1. Inspect by name similarity
            best_match = None
            best_score = 0.0
            
            for field, variations in self.target_fields.items():
                for var in variations:
                    score = self._jaro_winkler_sim(col, var)
                    if score > best_score and score > 0.82:
                        best_score = score
                        best_match = field
                        
            # 2. Inspect by data distribution heuristics if name is ambiguous
            if not best_match:
                best_match = self._guess_field_by_values(col, df[col])
                
            if best_match and best_match not in detected_mapping.values():
                detected_mapping[col] = best_match
                logger.info(f"[cyan]Mapped col '{col}' -> '{best_match}'[/cyan] (score/heuristic matched)")
                
        # Reverse mapping: standard_field -> dataset_col
        schema_mapping = {v: k for k, v in detected_mapping.items()}
        
        # Event order validation (checking if timestamp is sorted)
        ts_col = schema_mapping.get("timestamp")
        is_sorted = False
        if ts_col:
            try:
                ts_series = pd.to_datetime(df[ts_col])
                is_sorted = ts_series.is_monotonic_increasing
            except Exception:
                pass
                
        inspection_results = {
            "schema_mapping": schema_mapping,
            "missing_rates": missing_report,
            "event_ordering_sorted": is_sorted,
            "detected_fields": list(schema_mapping.keys()),
            "total_columns": list(df.columns)
        }
        
        logger.info(f"[green]Inspection complete. Discovered fields: {list(schema_mapping.keys())}[/green]")
        return inspection_results
