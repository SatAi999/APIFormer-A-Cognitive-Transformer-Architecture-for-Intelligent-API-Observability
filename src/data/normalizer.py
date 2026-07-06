import pandas as pd
from typing import List, Dict, Any
from src.data.synthetic_gen import UnifiedAPIEvent
from src.utils.logger import logger

class SchemaNormalizer:
    """Standardizes logs with heterogeneous schemas into the APIFormer+ Unified Event Format."""
    
    def __init__(self, inspection_results: Dict[str, Any]):
        self.mapping = inspection_results.get("schema_mapping", {})
        self.missing_rates = inspection_results.get("missing_rates", {})
        
    def normalize_dataframe(self, df: pd.DataFrame) -> List[UnifiedAPIEvent]:
        """Converts a pandas DataFrame into a list of standardized UnifiedAPIEvents."""
        normalized_events = []
        logger.info(f"Normalizing dataframe containing {len(df)} records...")
        
        # Ensure timestamp is parsed properly
        ts_col = self.mapping.get("timestamp")
        if ts_col:
            df = df.copy()
            df[ts_col] = pd.to_datetime(df[ts_col], errors='coerce')
            # Sort by timestamp to enforce chronological event ordering
            df = df.sort_values(by=ts_col)
            
        for _, row in df.iterrows():
            try:
                # Extracts mapped fields with clean fallbacks
                ts_val = row[ts_col].isoformat() if (ts_col and pd.notnull(row[ts_col])) else pd.Timestamp.now().isoformat()
                
                sess_col = self.mapping.get("session_id")
                sess_id = str(row[sess_col]) if (sess_col and pd.notnull(row[sess_col])) else "None"
                
                trace_col = self.mapping.get("trace_id")
                trace_id = str(row[trace_col]) if (trace_col and pd.notnull(row[trace_col])) else "None"
                
                corr_col = self.mapping.get("correlation_id")
                corr_id = str(row[corr_col]) if (corr_col and pd.notnull(row[corr_col])) else "None"
                
                user_col = self.mapping.get("user_id")
                user_id = str(row[user_col]) if (user_col and pd.notnull(row[user_col])) else "anonymous"
                
                service_col = self.mapping.get("service_name")
                service_name = str(row[service_col]) if (service_col and pd.notnull(row[service_col])) else "unknown-service"
                
                ep_col = self.mapping.get("endpoint")
                endpoint = str(row[ep_col]) if (ep_col and pd.notnull(row[ep_col])) else "/unknown"
                
                method_col = self.mapping.get("http_method")
                http_method = str(row[method_col]).upper() if (method_col and pd.notnull(row[method_col])) else "GET"
                
                status_col = self.mapping.get("status_code")
                try:
                    status_code = int(row[status_col]) if (status_col and pd.notnull(row[status_col])) else 200
                except (ValueError, TypeError):
                    status_code = 200
                    
                lat_col = self.mapping.get("latency")
                try:
                    latency_ms = float(row[lat_col]) if (lat_col and pd.notnull(row[lat_col])) else 50.0
                except (ValueError, TypeError):
                    latency_ms = 50.0
                    
                pay_col = self.mapping.get("payload_size")
                try:
                    payload_size_bytes = int(row[pay_col]) if (pay_col and pd.notnull(row[pay_col])) else 0
                except (ValueError, TypeError):
                    payload_size_bytes = 0
                
                # Check for dynamic fields or headers
                headers = str(row.get("request_headers", "None"))
                q_params = str(row.get("query_parameters", ""))
                device = str(row.get("device", "unknown"))
                geo = str(row.get("geo_location", "unknown"))
                auth = str(row.get("authentication", "None"))
                env = str(row.get("environment", "production"))
                event_type = str(row.get("event_type", "normal"))
                
                time_gap = float(row.get("time_since_previous_request", 0.0))
                
                event = UnifiedAPIEvent(
                    timestamp=ts_val,
                    session_id=sess_id,
                    trace_id=trace_id,
                    correlation_id=corr_id,
                    user_id=user_id,
                    service_name=service_name,
                    endpoint=endpoint,
                    http_method=http_method,
                    status_code=status_code,
                    latency_ms=latency_ms,
                    payload_size_bytes=payload_size_bytes,
                    request_headers=headers,
                    query_parameters=q_params,
                    device=device,
                    geo_location=geo,
                    authentication=auth,
                    environment=env,
                    event_type=event_type,
                    time_since_previous_request=time_gap
                )
                normalized_events.append(event)
            except Exception as e:
                logger.error(f"Error normalizing row: {e}")
                
        logger.info(f"Normalized {len(normalized_events)} events successfully.")
        return normalized_events
