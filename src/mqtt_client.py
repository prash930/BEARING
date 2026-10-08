import json
from datetime import datetime
import random
import time

class BearingMQTTClient:
    def __init__(self, broker='localhost', port=1883, topic='bearing/vibration'):
        self.broker = broker
        self.port = port
        self.topic = topic
        self.client = None
        self.callback = None
        self.connected = False
        self.latest_data = None

    def connect(self):
        try:
            import paho.mqtt.client as mqtt
            self.client = mqtt.Client()
            self.client.on_message = self.on_message
            self.client.connect(self.broker, self.port)
            self.connected = True
        except ImportError:
            print("Warning: paho-mqtt package not installed. MQTT client running in mock mode.")
        except Exception as e:
            print(f"Error connecting to MQTT broker: {e}")

    def disconnect(self):
        if self.client and self.connected:
            self.client.disconnect()
        self.connected = False

    def on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode('utf-8'))
            self.latest_data = {
                'bearing_id': payload.get('bearing_id'),
                'timestamp': payload.get('timestamp'),
                'ax': payload.get('ax'),
                'ay': payload.get('ay'),
                'az': payload.get('az'),
                'temperature': payload.get('temperature'),
                'current': payload.get('current'),
                'rpm': payload.get('rpm')
            }
            if self.callback:
                self.callback(self.latest_data)
        except Exception as e:
            print(f"Error parsing MQTT message: {e}")

    def start_listening(self, callback):
        self.callback = callback
        if self.client and self.connected:
            self.client.subscribe(self.topic)
            self.client.loop_start()

    def stop_listening(self):
        if self.client and self.connected:
            self.client.unsubscribe(self.topic)
            self.client.loop_stop()
        self.callback = None

    def get_latest_data(self):
        return self.latest_data

    def is_connected(self):
        return self.connected
        
    def simulate_data(self):
        """SIMULATED data generation for testing."""
        data = {
            "bearing_id": "B001",
            "timestamp": datetime.now().isoformat(),
            "ax": round(random.gauss(0.1, 0.05), 4),
            "ay": round(random.gauss(0.2, 0.05), 4),
            "az": round(random.gauss(1.0, 0.1), 4),
            "temperature": round(random.uniform(45.0, 65.0), 1),
            "current": round(random.uniform(1.2, 1.8), 2),
            "rpm": random.choice([1450, 2000])
        }
        self.latest_data = data
        if self.callback:
            self.callback(data)
        return data
