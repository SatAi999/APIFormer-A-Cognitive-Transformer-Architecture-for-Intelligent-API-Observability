import json
import time
import requests
from typing import Dict, Any, List, Optional

from src.utils.logger import logger

# Graceful import fallbacks for environments without Kafka or Redis running
try:
    from confluent_kafka import Consumer, Producer, KafkaError
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False
    
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


class APIFormerStreamingConsumer:
    """Consumes real-time API traffic from Kafka, groups events by session via Redis, 

    sends sequences to FastAPI serving app, and publishes alerts for detected anomalies.
    """
    
    def __init__(self, 
                 bootstrap_servers: str = "localhost:9092", 
                 group_id: str = "apiformer-ingest", 
                 redis_host: str = "localhost", 
                 redis_port: int = 6379,
                 predict_url: str = "http://localhost:8000/predict",
                 max_window: int = 128):
        self.predict_url = predict_url
        self.max_window = max_window
        
        # Initialize Redis Session Cache
        if REDIS_AVAILABLE:
            try:
                self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
                logger.info(f"Connected to Redis Session Cache at {redis_host}:{redis_port}")
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}. Falling back to in-memory cache.")
                self.redis_client = None
        else:
            self.redis_client = None
            
        self.memory_cache = {} # fallback in-memory cache
        
        # Initialize Kafka Broker connections
        if KAFKA_AVAILABLE:
            self.consumer = Consumer({
                'bootstrap.servers': bootstrap_servers,
                'group.id': group_id,
                'auto.offset.reset': 'earliest'
            })
            self.producer = Producer({
                'bootstrap.servers': bootstrap_servers
            })
            logger.info(f"Connected to Kafka brokers at {bootstrap_servers}")
        else:
            logger.warning("Confluent Kafka client not installed. Running in mock simulator mode.")
            self.consumer = None
            self.producer = None

    def get_session_sequence(self, session_id: str) -> List[Dict[str, Any]]:
        """Retrieves active sliding window request sequences from Redis or local cache."""
        if self.redis_client:
            try:
                data = self.redis_client.get(f"session:{session_id}")
                if data:
                    return json.loads(data)
            except Exception as e:
                logger.error(f"Redis get error: {e}")
        return self.memory_cache.get(session_id, [])

    def save_session_sequence(self, session_id: str, sequence: List[Dict[str, Any]]) -> None:
        """Saves updated request sequences back to Redis or local cache."""
        # Slice to sliding window constraint
        sequence = sequence[-self.max_window:]
        
        if self.redis_client:
            try:
                self.redis_client.setex(
                    f"session:{session_id}",
                    1800, # TTL 30 minutes
                    json.dumps(sequence)
                )
                return
            except Exception as e:
                logger.error(f"Redis set error: {e}")
        self.memory_cache[session_id] = sequence

    def process_message(self, message_value: str) -> Optional[Dict[str, Any]]:
        """Normalizes message, groups by session, sends to model, and handles alerts."""
        try:
            event = json.loads(message_value)
        except Exception as e:
            logger.error(f"Invalid message format: {e}")
            return None
            
        session_id = event.get("session_id", "anonymous")
        
        # Retrieve active session history and append new event
        sequence = self.get_session_sequence(session_id)
        sequence.append(event)
        self.save_session_sequence(session_id, sequence)
        
        # Query FastAPI Serving App for predictions
        payload = {"events": sequence}
        try:
            response = requests.post(self.predict_url, json=payload, timeout=2.0)
            if response.status_code == 200:
                res_data = response.json()
                
                # Check for anomalies
                predictions = res_data.get("predictions", {})
                step_anoms = predictions.get("step_anomalies", [])
                
                # Check if the latest step is anomalous
                if step_anoms:
                    latest_step = step_anoms[-1]
                    if latest_step.get("is_anomalous", False):
                        alert_msg = {
                            "session_id": session_id,
                            "timestamp": event.get("timestamp"),
                            "endpoint": latest_step.get("endpoint"),
                            "anomaly_probability": latest_step.get("anomaly_probability"),
                            "intent_intent": predictions.get("intent", {}).get("label"),
                            "is_bot": predictions.get("bot_detection", {}).get("is_bot")
                        }
                        
                        logger.error(f"[bold red]!!! SECURITY ALERT !!![/bold red] Anomaly detected in session {session_id} on {latest_step['endpoint']}")
                        self.publish_alert(alert_msg)
                        
                return res_data
        except requests.exceptions.RequestException as e:
            logger.warning(f"Failed to query model server: {e}")
            
        return None

    def publish_alert(self, alert_msg: Dict[str, Any]) -> None:
        """Publishes alert logs to api-alerts Kafka topic."""
        if self.producer:
            try:
                self.producer.produce(
                    "api-alerts",
                    key=alert_msg["session_id"].encode('utf-8'),
                    value=json.dumps(alert_msg).encode('utf-8')
                )
                self.producer.flush()
            except Exception as e:
                logger.error(f"Failed to produce Kafka alert message: {e}")

    def run(self, topic: str = "api-telemetry"):
        """Subscribes and runs the streaming consume loop."""
        if not self.consumer:
            logger.error("Consumer not initialized. Streaming loop cannot start.")
            return
            
        self.consumer.subscribe([topic])
        logger.info(f"Subscribed to topic: {topic}. Starting consume loop...")
        
        try:
            while True:
                msg = self.consumer.poll(1.0)
                if msg is None:
                    continue
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    else:
                        logger.error(f"Consumer error: {msg.error()}")
                        break
                        
                self.process_message(msg.value().decode('utf-8'))
        except KeyboardInterrupt:
            logger.info("Streaming consume loop stopped by user.")
        finally:
            self.consumer.close()
            
if __name__ == "__main__":
    consumer = APIFormerStreamingConsumer()
    # If running directly, start loop if kafka is available
    if KAFKA_AVAILABLE:
        consumer.run()
    else:
        print("Kafka not running locally. Simulator execution complete.")
