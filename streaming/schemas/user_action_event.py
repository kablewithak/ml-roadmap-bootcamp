"""
UserActionEvent Avro schema and Python dataclass.

Represents user actions for behavioral analysis and fraud detection.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from datetime import datetime
import uuid

# Avro schema definition for UserActionEvent
USER_ACTION_EVENT_SCHEMA = {
    "type": "record",
    "name": "UserActionEvent",
    "namespace": "com.streaming.events.user",
    "doc": "User action event for behavioral analysis and fraud detection",
    "fields": [
        {
            "name": "event_id",
            "type": "string",
            "doc": "Unique event identifier (UUID)",
            "logicalType": "uuid"
        },
        {
            "name": "user_id",
            "type": "string",
            "doc": "User identifier"
        },
        {
            "name": "session_id",
            "type": "string",
            "doc": "User session identifier"
        },
        {
            "name": "action_type",
            "type": {
                "type": "enum",
                "name": "ActionType",
                "symbols": [
                    "LOGIN",
                    "LOGOUT",
                    "PAGE_VIEW",
                    "BUTTON_CLICK",
                    "FORM_SUBMIT",
                    "SEARCH",
                    "ADD_TO_CART",
                    "REMOVE_FROM_CART",
                    "CHECKOUT_START",
                    "CHECKOUT_COMPLETE",
                    "PAYMENT_METHOD_ADD",
                    "PAYMENT_METHOD_UPDATE",
                    "PAYMENT_METHOD_DELETE",
                    "PROFILE_UPDATE",
                    "PASSWORD_CHANGE",
                    "TWO_FACTOR_ENABLE",
                    "TWO_FACTOR_DISABLE"
                ]
            },
            "doc": "Type of action performed"
        },
        {
            "name": "page_url",
            "type": ["null", "string"],
            "doc": "URL of the page where action occurred",
            "default": None
        },
        {
            "name": "referrer_url",
            "type": ["null", "string"],
            "doc": "Referrer URL",
            "default": None
        },
        {
            "name": "ip_address",
            "type": "string",
            "doc": "Client IP address"
        },
        {
            "name": "user_agent",
            "type": "string",
            "doc": "Browser user agent string"
        },
        {
            "name": "device_fingerprint",
            "type": ["null", "string"],
            "doc": "Device fingerprint for tracking",
            "default": None
        },
        {
            "name": "device_type",
            "type": {
                "type": "enum",
                "name": "DeviceType",
                "symbols": ["DESKTOP", "MOBILE", "TABLET", "UNKNOWN"]
            },
            "doc": "Type of device used",
            "default": "UNKNOWN"
        },
        {
            "name": "browser",
            "type": ["null", "string"],
            "doc": "Browser name and version",
            "default": None
        },
        {
            "name": "os",
            "type": ["null", "string"],
            "doc": "Operating system",
            "default": None
        },
        {
            "name": "country_code",
            "type": "string",
            "doc": "ISO 3166-1 alpha-2 country code"
        },
        {
            "name": "city",
            "type": ["null", "string"],
            "doc": "City name from IP geolocation",
            "default": None
        },
        {
            "name": "latitude",
            "type": ["null", "double"],
            "doc": "Latitude from IP geolocation",
            "default": None
        },
        {
            "name": "longitude",
            "type": ["null", "double"],
            "doc": "Longitude from IP geolocation",
            "default": None
        },
        {
            "name": "action_metadata",
            "type": ["null", {
                "type": "map",
                "values": "string"
            }],
            "doc": "Additional action-specific metadata",
            "default": None
        },
        {
            "name": "is_bot",
            "type": "boolean",
            "doc": "Whether the action appears to be from a bot",
            "default": False
        },
        {
            "name": "is_vpn",
            "type": "boolean",
            "doc": "Whether the IP appears to be from a VPN",
            "default": False
        },
        {
            "name": "is_tor",
            "type": "boolean",
            "doc": "Whether the IP is a known Tor exit node",
            "default": False
        },
        {
            "name": "time_on_page_ms",
            "type": ["null", "long"],
            "doc": "Time spent on page in milliseconds",
            "default": None
        },
        {
            "name": "idempotency_key",
            "type": "string",
            "doc": "Idempotency key for exactly-once processing"
        },
        {
            "name": "timestamp",
            "type": "long",
            "logicalType": "timestamp-millis",
            "doc": "Event timestamp in milliseconds since epoch"
        },
        {
            "name": "schema_version",
            "type": "string",
            "doc": "Schema version for evolution tracking",
            "default": "1.0.0"
        }
    ]
}


@dataclass
class UserActionEvent:
    """
    Python representation of UserActionEvent.

    This event captures user interactions with the system for behavioral
    analysis, fraud detection, and user experience optimization.

    Attributes:
        event_id: Unique event identifier
        user_id: User identifier
        session_id: Session identifier
        action_type: Type of action performed
        ip_address: Client IP address
        user_agent: Browser user agent
        country_code: ISO country code
        device_type: Type of device used
        idempotency_key: Key for exactly-once processing
        timestamp: Event timestamp
        page_url: URL where action occurred
        referrer_url: Referrer URL
        device_fingerprint: Device fingerprint
        browser: Browser name and version
        os: Operating system
        city: City from geolocation
        latitude: Latitude coordinate
        longitude: Longitude coordinate
        action_metadata: Additional metadata
        is_bot: Whether action is from a bot
        is_vpn: Whether IP is from VPN
        is_tor: Whether IP is Tor exit node
        time_on_page_ms: Time spent on page
        schema_version: Schema version
    """

    event_id: str
    user_id: str
    session_id: str
    action_type: str
    ip_address: str
    user_agent: str
    country_code: str
    device_type: str
    idempotency_key: str
    timestamp: int
    page_url: Optional[str] = None
    referrer_url: Optional[str] = None
    device_fingerprint: Optional[str] = None
    browser: Optional[str] = None
    os: Optional[str] = None
    city: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    action_metadata: Optional[Dict[str, str]] = field(default_factory=dict)
    is_bot: bool = False
    is_vpn: bool = False
    is_tor: bool = False
    time_on_page_ms: Optional[int] = None
    schema_version: str = "1.0.0"

    @classmethod
    def create(
        cls,
        user_id: str,
        session_id: str,
        action_type: str,
        ip_address: str,
        user_agent: str,
        country_code: str,
        device_type: str = "UNKNOWN",
        **kwargs
    ) -> 'UserActionEvent':
        """
        Factory method to create a UserActionEvent.

        Args:
            user_id: User identifier
            session_id: Session identifier
            action_type: Type of action
            ip_address: Client IP
            user_agent: User agent string
            country_code: ISO country code
            device_type: Type of device
            **kwargs: Additional optional fields

        Returns:
            New UserActionEvent instance
        """
        event_id = str(uuid.uuid4())
        idempotency_key = kwargs.pop('idempotency_key', f"{user_id}:{event_id}")
        timestamp = kwargs.pop('timestamp', int(datetime.utcnow().timestamp() * 1000))

        return cls(
            event_id=event_id,
            user_id=user_id,
            session_id=session_id,
            action_type=action_type,
            ip_address=ip_address,
            user_agent=user_agent,
            country_code=country_code,
            device_type=device_type,
            idempotency_key=idempotency_key,
            timestamp=timestamp,
            **kwargs
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Avro serialization."""
        return {
            'event_id': self.event_id,
            'user_id': self.user_id,
            'session_id': self.session_id,
            'action_type': self.action_type,
            'page_url': self.page_url,
            'referrer_url': self.referrer_url,
            'ip_address': self.ip_address,
            'user_agent': self.user_agent,
            'device_fingerprint': self.device_fingerprint,
            'device_type': self.device_type,
            'browser': self.browser,
            'os': self.os,
            'country_code': self.country_code,
            'city': self.city,
            'latitude': self.latitude,
            'longitude': self.longitude,
            'action_metadata': self.action_metadata,
            'is_bot': self.is_bot,
            'is_vpn': self.is_vpn,
            'is_tor': self.is_tor,
            'time_on_page_ms': self.time_on_page_ms,
            'idempotency_key': self.idempotency_key,
            'timestamp': self.timestamp,
            'schema_version': self.schema_version
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserActionEvent':
        """Create UserActionEvent from dictionary."""
        return cls(**data)

    def is_suspicious(self) -> bool:
        """
        Determine if the action has suspicious characteristics.

        Returns:
            True if action shows suspicious patterns
        """
        return self.is_bot or self.is_tor or (
            self.is_vpn and self.action_type in [
                'LOGIN', 'PAYMENT_METHOD_ADD', 'CHECKOUT_COMPLETE'
            ]
        )

    def is_authentication_action(self) -> bool:
        """Check if this is an authentication-related action."""
        return self.action_type in [
            'LOGIN', 'LOGOUT', 'PASSWORD_CHANGE',
            'TWO_FACTOR_ENABLE', 'TWO_FACTOR_DISABLE'
        ]

    def is_payment_action(self) -> bool:
        """Check if this is a payment-related action."""
        return self.action_type in [
            'CHECKOUT_START', 'CHECKOUT_COMPLETE',
            'PAYMENT_METHOD_ADD', 'PAYMENT_METHOD_UPDATE',
            'PAYMENT_METHOD_DELETE'
        ]
