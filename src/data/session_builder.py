from datetime import datetime
from typing import List, Dict, Any
from collections import defaultdict
from src.data.synthetic_gen import UnifiedAPIEvent
from src.utils.logger import logger

class SessionBuilder:
    """Groups API events into ordered session sequences using trace/session keys or time window heuristics."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get("session_builder", {})
        self.idle_timeout = self.config.get("idle_timeout_seconds", 1800)
        self.max_len = self.config.get("max_sequence_length", 128)
        
    def reconstruct_sessions(self, events: List[UnifiedAPIEvent]) -> Dict[str, List[UnifiedAPIEvent]]:
        """Groups list of events into sessions, sorting by timestamp, and calculates temporal time gaps."""
        logger.info(f"Reconstructing sessions from {len(events)} events...")
        
        # Sort events by timestamp to ensure chronological order
        events = sorted(events, key=lambda e: e.timestamp)
        
        # Check if we have explicit sessions
        has_session_id = any(e.session_id != "None" for e in events)
        
        sessions = defaultdict(list)
        
        if has_session_id:
            # Group directly by session ID
            for e in events:
                sess_key = e.session_id if e.session_id != "None" else "orphan_events"
                sessions[sess_key].append(e)
            logger.info(f"Grouped events into {len(sessions)} sessions using explicit session IDs.")
        else:
            # Group by user_id and temporal windowing
            user_events = defaultdict(list)
            for e in events:
                user_events[e.user_id].append(e)
                
            session_counter = 0
            for user_id, u_events in user_events.items():
                # Sort user events chronologically
                u_events = sorted(u_events, key=lambda e: e.timestamp)
                
                current_session_id = f"sess_user_{user_id}_{session_counter}"
                prev_time = None
                
                for e in u_events:
                    curr_time = datetime.fromisoformat(e.timestamp)
                    if prev_time is not None:
                        delta = (curr_time - prev_time).total_seconds()
                        if delta > self.idle_timeout:
                            # Start a new session if user was idle longer than threshold
                            session_counter += 1
                            current_session_id = f"sess_user_{user_id}_{session_counter}"
                            
                    e.session_id = current_session_id
                    sessions[current_session_id].append(e)
                    prev_time = curr_time
            logger.info(f"Grouped events into {len(sessions)} sessions using user-time windowing ({self.idle_timeout}s timeout).")
            
        # Post-process sessions: sort, calculate correct inter-request delays, and segment to max sequence length
        final_sessions = {}
        for s_id, s_events in sessions.items():
            if s_id == "orphan_events":
                continue
                
            # Sort events in the session
            s_events = sorted(s_events, key=lambda e: e.timestamp)
            
            # Recalculate time gaps
            prev_t = None
            for e in s_events:
                curr_t = datetime.fromisoformat(e.timestamp)
                if prev_t is None:
                    e.time_since_previous_request = 0.0
                else:
                    e.time_since_previous_request = max(0.0, (curr_t - prev_t).total_seconds())
                prev_t = curr_t
                
            # Split sessions longer than max_len into sub-sessions
            if len(s_events) > self.max_len:
                sub_count = 0
                for chunk_idx in range(0, len(s_events), self.max_len):
                    sub_sess_id = f"{s_id}_sub_{sub_count}"
                    chunk = s_events[chunk_idx:chunk_idx + self.max_len]
                    # Reset the first element's delay in the new chunk to 0.0
                    chunk[0].time_since_previous_request = 0.0
                    # Re-assign session id
                    for e in chunk:
                        e.session_id = sub_sess_id
                    final_sessions[sub_sess_id] = chunk
                    sub_count += 1
            else:
                final_sessions[s_id] = s_events
                
        logger.info(f"Final session count after size segmentation (max_len={self.max_len}): {len(final_sessions)}")
        return final_sessions
