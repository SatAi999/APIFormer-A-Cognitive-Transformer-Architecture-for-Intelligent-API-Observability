import uuid
from datetime import datetime, timedelta
import random
import numpy as np
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, asdict

@dataclass
class UnifiedAPIEvent:
    timestamp: str
    session_id: str
    trace_id: str
    correlation_id: str
    user_id: str
    service_name: str
    endpoint: str
    http_method: str
    status_code: int
    latency_ms: float
    payload_size_bytes: int
    request_headers: str
    query_parameters: str
    device: str
    geo_location: str
    authentication: str
    environment: str
    event_type: str  # e.g., 'normal', '503_cascade', 'credential_stuffing', etc.
    time_since_previous_request: float

class SyntheticAPIGenerator:
    """Generates realistic enterprise-scale API sessions for APIFormer+ pretraining."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.generator_cfg = config.get("generator", {})
        self.seed = self.generator_cfg.get("seed", 42)
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        # Latency Parameters
        self.mu = self.generator_cfg.get("latency", {}).get("mu", 4.8)
        self.sigma = self.generator_cfg.get("latency", {}).get("sigma", 0.6)
        self.timeout = self.generator_cfg.get("latency", {}).get("timeout_threshold", 5000.0)
        
        # Services & Endpoints configuration
        self.services = {
            "gateway": ["/api/v1/auth/login", "/api/v1/products", "/api/v1/products/{id}", "/api/v1/cart", "/api/v1/checkout", "/api/v1/admin/dashboard", "/api/v1/admin/users", "/api/v1/admin/logs"],
            "auth-service": ["/internal/v1/auth/validate", "/internal/v1/auth/login"],
            "catalog-service": ["/internal/v1/products", "/internal/v1/products/{id}"],
            "cart-service": ["/internal/v1/cart/add", "/internal/v1/cart/view"],
            "payment-service": ["/internal/v1/payments/charge", "/internal/v1/payments/refund"],
            "invoice-service": ["/internal/v1/invoices/generate"],
        }
        
        self.devices = ["desktop-chrome", "desktop-safari", "mobile-ios", "mobile-android", "bot-agent"]
        self.locations = ["US-East", "US-West", "EU-Central", "AP-South", "CN-East"]
        self.environments = ["production", "staging"]
        
    def _generate_latency(self, factor: float = 1.0, is_timeout: bool = False) -> float:
        if is_timeout:
            return float(self.timeout + random.uniform(50, 500))
        # Log-normal distribution
        val = np.random.lognormal(mean=self.mu, sigma=self.sigma) * factor
        return float(min(val, self.timeout))

    def _generate_headers(self, user_agent: str, token: str = "None") -> str:
        return f"User-Agent: {user_agent} | Accept: application/json | Authorization: {token}"

    def generate_normal_session(self, start_time: datetime) -> List[UnifiedAPIEvent]:
        """Simulates a typical customer e-commerce journey."""
        session_id = str(uuid.uuid4())
        user_id = f"user_{random.randint(1000, 9999)}"
        device = random.choice(self.devices[:-1]) # exclude bot
        location = random.choice(self.locations)
        env = random.choice(self.environments)
        auth_token = f"Bearer {uuid.uuid4().hex[:30]}"
        
        events = []
        curr_time = start_time
        
        # Sequence: Login -> Browse -> View Product -> Add to Cart -> Checkout -> Pay -> Invoice -> Logout
        sequence = [
            ("gateway", "/api/v1/auth/login", "POST", "None", 200, 1.0),
            ("gateway", "/api/v1/products", "GET", auth_token, 200, 1.2),
            ("gateway", "/api/v1/products/{id}", "GET", auth_token, 200, 1.1),
            ("gateway", "/api/v1/cart", "POST", auth_token, 200, 1.3),
            ("gateway", "/api/v1/checkout", "POST", auth_token, 200, 1.5),
            ("gateway", "/api/v1/invoices/generate", "POST", auth_token, 201, 1.4) # payment happens internal to checkout
        ]
        
        prev_time = curr_time
        for service, endpoint, method, auth, status, lat_factor in sequence:
            # Internal service chain simulation (creating multi-service traces)
            trace_id = str(uuid.uuid4())
            corr_id = str(uuid.uuid4())
            
            # Microservice hop emulation
            # gateway request
            lat = self._generate_latency(lat_factor)
            pay_size = random.randint(200, 1500)
            delay = (curr_time - prev_time).total_seconds()
            
            events.append(UnifiedAPIEvent(
                timestamp=curr_time.isoformat(),
                session_id=session_id,
                trace_id=trace_id,
                correlation_id=corr_id,
                user_id=user_id,
                service_name=service,
                endpoint=endpoint,
                http_method=method,
                status_code=status,
                latency_ms=lat,
                payload_size_bytes=pay_size,
                request_headers=self._generate_headers(device, auth),
                query_parameters="limit=10" if "products" in endpoint else "",
                device=device,
                geo_location=location,
                authentication="Bearer" if auth != "None" else "None",
                environment=env,
                event_type="normal",
                time_since_previous_request=delay
            ))
            
            # Simulating internal service call as part of the trace
            if endpoint == "/api/v1/auth/login":
                events.append(UnifiedAPIEvent(
                    timestamp=(curr_time + timedelta(milliseconds=10)).isoformat(),
                    session_id=session_id,
                    trace_id=trace_id,
                    correlation_id=corr_id,
                    user_id=user_id,
                    service_name="auth-service",
                    endpoint="/internal/v1/auth/login",
                    http_method="POST",
                    status_code=200,
                    latency_ms=lat * 0.4,
                    payload_size_bytes=pay_size // 2,
                    request_headers=self._generate_headers("internal-caller", auth),
                    query_parameters="",
                    device="internal",
                    geo_location=location,
                    authentication="Internal-Key",
                    environment=env,
                    event_type="normal",
                    time_since_previous_request=0.01
                ))
            elif endpoint == "/api/v1/checkout":
                # checkout calls payment-service
                events.append(UnifiedAPIEvent(
                    timestamp=(curr_time + timedelta(milliseconds=20)).isoformat(),
                    session_id=session_id,
                    trace_id=trace_id,
                    correlation_id=corr_id,
                    user_id=user_id,
                    service_name="payment-service",
                    endpoint="/internal/v1/payments/charge",
                    http_method="POST",
                    status_code=200,
                    latency_ms=lat * 0.7,
                    payload_size_bytes=400,
                    request_headers=self._generate_headers("internal-caller", auth),
                    query_parameters="",
                    device="internal",
                    geo_location=location,
                    authentication="Internal-Key",
                    environment=env,
                    event_type="normal",
                    time_since_previous_request=0.02
                ))
            
            prev_time = curr_time
            curr_time += timedelta(seconds=random.uniform(1.0, 15.0))
            
        return events

    def generate_503_cascade_session(self, start_time: datetime) -> List[UnifiedAPIEvent]:
        """Simulates internal dependency failure cascading up the stack."""
        session_id = str(uuid.uuid4())
        user_id = f"user_{random.randint(1000, 9999)}"
        device = random.choice(self.devices[:-1])
        location = random.choice(self.locations)
        env = random.choice(self.environments)
        auth_token = f"Bearer {uuid.uuid4().hex[:30]}"
        
        events = []
        curr_time = start_time
        
        # Journey goes fine until Checkout, which triggers payment failure
        sequence = [
            ("gateway", "/api/v1/auth/login", "POST", "None", 200, 1.0),
            ("gateway", "/api/v1/products", "GET", auth_token, 200, 1.2),
            ("gateway", "/api/v1/products/{id}", "GET", auth_token, 200, 1.1),
            ("gateway", "/api/v1/cart", "POST", auth_token, 200, 1.3),
            ("gateway", "/api/v1/checkout", "POST", auth_token, 503, 10.0), # Latency spike!
        ]
        
        prev_time = curr_time
        for service, endpoint, method, auth, status, lat_factor in sequence:
            trace_id = str(uuid.uuid4())
            corr_id = str(uuid.uuid4())
            delay = (curr_time - prev_time).total_seconds()
            
            if status == 503:
                # Payment service fails, causing a timeout and 500/503 cascade
                lat = self._generate_latency(lat_factor, is_timeout=True)
                events.append(UnifiedAPIEvent(
                    timestamp=curr_time.isoformat(),
                    session_id=session_id,
                    trace_id=trace_id,
                    correlation_id=corr_id,
                    user_id=user_id,
                    service_name=service,
                    endpoint=endpoint,
                    http_method=method,
                    status_code=503,
                    latency_ms=lat,
                    payload_size_bytes=150,
                    request_headers=self._generate_headers(device, auth),
                    query_parameters="",
                    device=device,
                    geo_location=location,
                    authentication="Bearer",
                    environment=env,
                    event_type="503_cascade",
                    time_since_previous_request=delay
                ))
                
                # Internal service call failures (payment-service throws 500 due to db outage)
                events.append(UnifiedAPIEvent(
                    timestamp=(curr_time + timedelta(milliseconds=15)).isoformat(),
                    session_id=session_id,
                    trace_id=trace_id,
                    correlation_id=corr_id,
                    user_id=user_id,
                    service_name="payment-service",
                    endpoint="/internal/v1/payments/charge",
                    http_method="POST",
                    status_code=500,
                    latency_ms=lat * 0.9,
                    payload_size_bytes=50,
                    request_headers=self._generate_headers("internal-caller", auth),
                    query_parameters="",
                    device="internal",
                    geo_location=location,
                    authentication="Internal-Key",
                    environment=env,
                    event_type="503_cascade",
                    time_since_previous_request=0.015
                ))
            else:
                lat = self._generate_latency(lat_factor)
                events.append(UnifiedAPIEvent(
                    timestamp=curr_time.isoformat(),
                    session_id=session_id,
                    trace_id=trace_id,
                    correlation_id=corr_id,
                    user_id=user_id,
                    service_name=service,
                    endpoint=endpoint,
                    http_method=method,
                    status_code=status,
                    latency_ms=lat,
                    payload_size_bytes=random.randint(200, 1500),
                    request_headers=self._generate_headers(device, auth),
                    query_parameters="",
                    device=device,
                    geo_location=location,
                    authentication="Bearer" if auth != "None" else "None",
                    environment=env,
                    event_type="normal",
                    time_since_previous_request=delay
                ))
                
            prev_time = curr_time
            curr_time += timedelta(seconds=random.uniform(1.0, 5.0))
            
        return events

    def generate_credential_stuffing_session(self, start_time: datetime) -> List[UnifiedAPIEvent]:
        """Simulates rapid, unauthorized login attempts (Credential Stuffing / Bot attack)."""
        session_id = str(uuid.uuid4())
        device = "bot-agent"
        location = random.choice(self.locations)
        env = "production"
        
        events = []
        curr_time = start_time
        prev_time = curr_time
        
        # Hit /api/v1/auth/login repeatedly at high frequencies
        num_attempts = random.randint(20, 50)
        for _ in range(num_attempts):
            trace_id = str(uuid.uuid4())
            corr_id = str(uuid.uuid4())
            delay = (curr_time - prev_time).total_seconds()
            
            # High speed, failed status (401 Unauthorized), low latency because auth catches it fast
            lat = self._generate_latency(0.3)
            events.append(UnifiedAPIEvent(
                timestamp=curr_time.isoformat(),
                session_id=session_id,
                trace_id=trace_id,
                correlation_id=corr_id,
                user_id="anonymous",
                service_name="gateway",
                endpoint="/api/v1/auth/login",
                http_method="POST",
                status_code=401,
                latency_ms=lat,
                payload_size_bytes=random.randint(120, 150),
                request_headers=self._generate_headers(device, "None"),
                query_parameters="",
                device=device,
                geo_location=location,
                authentication="None",
                environment=env,
                event_type="credential_stuffing",
                time_since_previous_request=delay
            ))
            
            prev_time = curr_time
            # Sleep milliseconds to simulate rapid bot traffic
            curr_time += timedelta(milliseconds=random.randint(30, 150))
            
        return events

    def generate_bot_scraping_session(self, start_time: datetime) -> List[UnifiedAPIEvent]:
        """Simulates rapid, serial product page scraping."""
        session_id = str(uuid.uuid4())
        device = "bot-agent"
        location = random.choice(self.locations)
        env = "production"
        
        events = []
        curr_time = start_time
        prev_time = curr_time
        
        # Scrape /api/v1/products/{id} in rapid succession
        num_requests = random.randint(40, 100)
        for i in range(num_requests):
            trace_id = str(uuid.uuid4())
            corr_id = str(uuid.uuid4())
            delay = (curr_time - prev_time).total_seconds()
            
            # Systematic, regular time delays (e.g. exactly 200ms +/- 10ms)
            lat = self._generate_latency(0.8)
            events.append(UnifiedAPIEvent(
                timestamp=curr_time.isoformat(),
                session_id=session_id,
                trace_id=trace_id,
                correlation_id=corr_id,
                user_id="anonymous",
                service_name="gateway",
                endpoint="/api/v1/products/{id}",
                http_method="GET",
                status_code=200,
                latency_ms=lat,
                payload_size_bytes=random.randint(5000, 12000), # large scraping response
                request_headers=self._generate_headers(device, "None"),
                query_parameters=f"id={random.randint(10000, 99999)}",
                device=device,
                geo_location=location,
                authentication="None",
                environment=env,
                event_type="bot_scraping",
                time_since_previous_request=delay
            ))
            
            prev_time = curr_time
            curr_time += timedelta(milliseconds=random.randint(180, 220))
            
        return events

    def generate_sequence_abuse_session(self, start_time: datetime) -> List[UnifiedAPIEvent]:
        """Simulates workflow violation (calling payment/checkout directly without auth/cart)."""
        session_id = str(uuid.uuid4())
        user_id = "malicious_user"
        device = random.choice(self.devices[:-1])
        location = random.choice(self.locations)
        env = "production"
        
        events = []
        curr_time = start_time
        prev_time = curr_time
        
        # Directly target checkout and billing endpoints
        sequence = [
            ("gateway", "/api/v1/checkout", "POST", "None", 403, 0.4), # Access denied!
            ("gateway", "/api/v1/checkout", "POST", "None", 403, 0.4),
            ("gateway", "/api/v1/admin/dashboard", "GET", "None", 401, 0.3),
        ]
        
        for service, endpoint, method, auth, status, lat_factor in sequence:
            trace_id = str(uuid.uuid4())
            corr_id = str(uuid.uuid4())
            delay = (curr_time - prev_time).total_seconds()
            
            lat = self._generate_latency(lat_factor)
            events.append(UnifiedAPIEvent(
                timestamp=curr_time.isoformat(),
                session_id=session_id,
                trace_id=trace_id,
                correlation_id=corr_id,
                user_id=user_id,
                service_name=service,
                endpoint=endpoint,
                http_method=method,
                status_code=status,
                latency_ms=lat,
                payload_size_bytes=random.randint(100, 300),
                request_headers=self._generate_headers(device, auth),
                query_parameters="",
                device=device,
                geo_location=location,
                authentication="None",
                environment=env,
                event_type="sequence_abuse",
                time_since_previous_request=delay
            ))
            
            prev_time = curr_time
            curr_time += timedelta(seconds=random.uniform(0.5, 3.0))
            
        return events

    def generate_dataset(self) -> List[UnifiedAPIEvent]:
        """Generates thousands of sessions spanning normal and various abnormal traffic states."""
        num_sessions = self.generator_cfg.get("num_sessions", 100)
        normal_ratio = self.generator_cfg.get("normal_ratio", 0.8)
        
        all_events = []
        start_time = datetime.now() - timedelta(days=1)
        
        for i in range(num_sessions):
            # Shift time to generate sequential transactions over time
            session_start = start_time + timedelta(minutes=i * random.uniform(1.0, 5.0))
            
            # Roll for normal vs anomaly
            if random.random() < normal_ratio:
                session_events = self.generate_normal_session(session_start)
            else:
                # Anomaly selection
                anomaly_type = random.choice(["503_cascade", "credential_stuffing", "bot_scraping", "sequence_abuse"])
                if anomaly_type == "503_cascade":
                    session_events = self.generate_503_cascade_session(session_start)
                elif anomaly_type == "credential_stuffing":
                    session_events = self.generate_credential_stuffing_session(session_start)
                elif anomaly_type == "bot_scraping":
                    session_events = self.generate_bot_scraping_session(session_start)
                else:
                    session_events = self.generate_sequence_abuse_session(session_start)
            
            all_events.extend(session_events)
            
        # Sort all events globally by timestamp to represent true enterprise logs arriving chronologically
        all_events.sort(key=lambda x: x.timestamp)
        return all_events
