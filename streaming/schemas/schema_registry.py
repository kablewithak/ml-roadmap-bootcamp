"""
Schema Registry integration and schema evolution management.

Provides utilities for working with Confluent Schema Registry (compatible with Redpanda),
including schema registration, compatibility checking, and evolution strategies.
"""

import logging
from typing import Dict, Any, Optional, List
from enum import Enum
import requests
import json

logger = logging.getLogger(__name__)


class CompatibilityMode(Enum):
    """
    Schema compatibility modes.

    BACKWARD: New schema can read data written with old schema
              (consumers updated before producers)
    FORWARD: Old schema can read data written with new schema
             (producers updated before consumers)
    FULL: Both backward and forward compatible
    BACKWARD_TRANSITIVE: Backward compatible with all previous versions
    FORWARD_TRANSITIVE: Forward compatible with all previous versions
    FULL_TRANSITIVE: Both backward and forward with all versions
    NONE: No compatibility checking
    """
    BACKWARD = "BACKWARD"
    FORWARD = "FORWARD"
    FULL = "FULL"
    BACKWARD_TRANSITIVE = "BACKWARD_TRANSITIVE"
    FORWARD_TRANSITIVE = "FORWARD_TRANSITIVE"
    FULL_TRANSITIVE = "FULL_TRANSITIVE"
    NONE = "NONE"


class SchemaRegistryClient:
    """
    Client for interacting with Confluent Schema Registry.

    Provides methods for registering schemas, fetching schemas by ID or subject,
    and managing schema versions.

    Attributes:
        base_url: Schema Registry base URL
        headers: HTTP headers for requests
    """

    def __init__(self, base_url: str = "http://localhost:18081"):
        """
        Initialize Schema Registry client.

        Args:
            base_url: Schema Registry URL (default: http://localhost:18081)
        """
        self.base_url = base_url.rstrip('/')
        self.headers = {'Content-Type': 'application/vnd.schemaregistry.v1+json'}
        logger.info(f"Initialized Schema Registry client: {self.base_url}")

    def register_schema(
        self,
        subject: str,
        schema: Dict[str, Any],
        schema_type: str = "AVRO"
    ) -> int:
        """
        Register a schema with the Schema Registry.

        Args:
            subject: Subject name (typically: topic-key or topic-value)
            schema: Avro schema definition
            schema_type: Schema type (AVRO, JSON, PROTOBUF)

        Returns:
            Schema ID assigned by the registry

        Raises:
            requests.HTTPError: If registration fails
        """
        url = f"{self.base_url}/subjects/{subject}/versions"
        payload = {
            "schema": json.dumps(schema),
            "schemaType": schema_type
        }

        logger.info(f"Registering schema for subject: {subject}")
        response = requests.post(url, json=payload, headers=self.headers)
        response.raise_for_status()

        schema_id = response.json()['id']
        logger.info(f"Schema registered successfully. ID: {schema_id}")
        return schema_id

    def get_schema_by_id(self, schema_id: int) -> Dict[str, Any]:
        """
        Retrieve schema by ID.

        Args:
            schema_id: Schema ID

        Returns:
            Schema definition

        Raises:
            requests.HTTPError: If schema not found
        """
        url = f"{self.base_url}/schemas/ids/{schema_id}"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()

        schema_str = response.json()['schema']
        return json.loads(schema_str)

    def get_latest_schema(self, subject: str) -> tuple[Dict[str, Any], int]:
        """
        Get the latest schema version for a subject.

        Args:
            subject: Subject name

        Returns:
            Tuple of (schema definition, version number)

        Raises:
            requests.HTTPError: If subject not found
        """
        url = f"{self.base_url}/subjects/{subject}/versions/latest"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()

        data = response.json()
        schema = json.loads(data['schema'])
        version = data['version']

        logger.info(f"Retrieved latest schema for {subject}, version: {version}")
        return schema, version

    def get_all_subjects(self) -> List[str]:
        """
        Get all registered subjects.

        Returns:
            List of subject names
        """
        url = f"{self.base_url}/subjects"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()

    def get_subject_versions(self, subject: str) -> List[int]:
        """
        Get all version numbers for a subject.

        Args:
            subject: Subject name

        Returns:
            List of version numbers
        """
        url = f"{self.base_url}/subjects/{subject}/versions"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()

    def check_compatibility(
        self,
        subject: str,
        schema: Dict[str, Any],
        version: str = "latest"
    ) -> bool:
        """
        Check if a schema is compatible with a specific version.

        Args:
            subject: Subject name
            schema: Schema to check
            version: Version to check against (default: "latest")

        Returns:
            True if compatible, False otherwise
        """
        url = f"{self.base_url}/compatibility/subjects/{subject}/versions/{version}"
        payload = {"schema": json.dumps(schema)}

        response = requests.post(url, json=payload, headers=self.headers)
        response.raise_for_status()

        is_compatible = response.json().get('is_compatible', False)
        logger.info(f"Compatibility check for {subject}: {is_compatible}")
        return is_compatible

    def set_compatibility(
        self,
        subject: str,
        compatibility: CompatibilityMode
    ) -> None:
        """
        Set compatibility mode for a subject.

        Args:
            subject: Subject name
            compatibility: Compatibility mode
        """
        url = f"{self.base_url}/config/{subject}"
        payload = {"compatibility": compatibility.value}

        response = requests.put(url, json=payload, headers=self.headers)
        response.raise_for_status()

        logger.info(f"Set compatibility for {subject} to {compatibility.value}")

    def get_compatibility(self, subject: str) -> CompatibilityMode:
        """
        Get compatibility mode for a subject.

        Args:
            subject: Subject name

        Returns:
            Current compatibility mode
        """
        url = f"{self.base_url}/config/{subject}"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()

        mode = response.json()['compatibilityLevel']
        return CompatibilityMode(mode)

    def delete_subject(self, subject: str, permanent: bool = False) -> List[int]:
        """
        Delete a subject (soft or hard delete).

        Args:
            subject: Subject name
            permanent: If True, permanently delete (default: False for soft delete)

        Returns:
            List of deleted version numbers
        """
        url = f"{self.base_url}/subjects/{subject}"
        if permanent:
            url += "?permanent=true"

        response = requests.delete(url, headers=self.headers)
        response.raise_for_status()

        logger.warning(f"Deleted subject: {subject} (permanent: {permanent})")
        return response.json()


class SchemaEvolutionManager:
    """
    Manages schema evolution with best practices for production systems.

    Handles schema registration, compatibility checking, and provides
    strategies for evolving schemas over time while maintaining backward
    compatibility.
    """

    def __init__(self, schema_registry_url: str = "http://localhost:18081"):
        """
        Initialize Schema Evolution Manager.

        Args:
            schema_registry_url: Schema Registry URL
        """
        self.client = SchemaRegistryClient(schema_registry_url)
        logger.info("Initialized Schema Evolution Manager")

    def register_with_compatibility_check(
        self,
        subject: str,
        schema: Dict[str, Any],
        compatibility_mode: CompatibilityMode = CompatibilityMode.BACKWARD_TRANSITIVE
    ) -> int:
        """
        Register a schema after ensuring compatibility.

        Production strategy:
        1. Start with BACKWARD compatibility (new code reads old data)
        2. Move to FULL when you need bidirectional compatibility
        3. Use BACKWARD_TRANSITIVE for production systems

        Args:
            subject: Subject name
            schema: Schema definition
            compatibility_mode: Desired compatibility mode

        Returns:
            Schema ID

        Raises:
            ValueError: If schema is not compatible with existing versions
        """
        # Set compatibility mode first
        try:
            self.client.set_compatibility(subject, compatibility_mode)
        except requests.HTTPError:
            # Subject doesn't exist yet, that's okay
            pass

        # Check if subject has existing versions
        try:
            versions = self.client.get_subject_versions(subject)
            if versions:
                # Check compatibility with latest version
                is_compatible = self.client.check_compatibility(subject, schema)
                if not is_compatible:
                    raise ValueError(
                        f"Schema for {subject} is not compatible with existing "
                        f"versions under {compatibility_mode.value} mode"
                    )
        except requests.HTTPError as e:
            # Subject doesn't exist, first registration
            logger.info(f"First schema registration for {subject}")

        # Register the schema
        return self.client.register_schema(subject, schema)

    def evolve_schema_safely(
        self,
        subject: str,
        old_schema: Dict[str, Any],
        new_schema: Dict[str, Any]
    ) -> tuple[bool, List[str]]:
        """
        Analyze schema evolution and provide safety recommendations.

        Args:
            subject: Subject name
            old_schema: Current schema
            new_schema: Proposed new schema

        Returns:
            Tuple of (is_safe, list of warnings/recommendations)
        """
        warnings = []
        is_safe = True

        old_fields = {f['name']: f for f in old_schema.get('fields', [])}
        new_fields = {f['name']: f for f in new_schema.get('fields', [])}

        # Check for removed fields (breaking change)
        removed_fields = set(old_fields.keys()) - set(new_fields.keys())
        if removed_fields:
            is_safe = False
            warnings.append(
                f"BREAKING: Removed fields: {removed_fields}. "
                "This will break old consumers."
            )

        # Check for added required fields without defaults (breaking change)
        for field_name, field_def in new_fields.items():
            if field_name not in old_fields:
                if 'default' not in field_def and field_def['type'] != ['null', 'string']:
                    is_safe = False
                    warnings.append(
                        f"BREAKING: Added required field '{field_name}' without default. "
                        "Old producers cannot write this field."
                    )

        # Check for type changes (potentially breaking)
        for field_name in set(old_fields.keys()) & set(new_fields.keys()):
            old_type = old_fields[field_name]['type']
            new_type = new_fields[field_name]['type']

            if old_type != new_type:
                warnings.append(
                    f"WARNING: Type changed for field '{field_name}': "
                    f"{old_type} -> {new_type}. Verify compatibility."
                )

        # Best practices
        if not warnings:
            warnings.append(
                "Schema evolution looks safe. Remember to:\n"
                "1. Deploy consumers first (they can read old data)\n"
                "2. Then deploy producers (they write new data)"
            )

        return is_safe, warnings

    def register_event_schemas(
        self,
        topic: str,
        key_schema: Optional[Dict[str, Any]],
        value_schema: Dict[str, Any]
    ) -> tuple[Optional[int], int]:
        """
        Register both key and value schemas for a topic.

        Convention: Use subject names like "topic-key" and "topic-value"

        Args:
            topic: Topic name
            key_schema: Key schema (None for string keys)
            value_schema: Value schema

        Returns:
            Tuple of (key_schema_id, value_schema_id)
        """
        key_schema_id = None
        if key_schema:
            key_subject = f"{topic}-key"
            key_schema_id = self.register_with_compatibility_check(
                key_subject, key_schema
            )

        value_subject = f"{topic}-value"
        value_schema_id = self.register_with_compatibility_check(
            value_subject, value_schema
        )

        logger.info(
            f"Registered schemas for topic {topic}: "
            f"key_id={key_schema_id}, value_id={value_schema_id}"
        )

        return key_schema_id, value_schema_id

    def get_schema_lineage(self, subject: str) -> List[Dict[str, Any]]:
        """
        Get the evolution history of a schema.

        Args:
            subject: Subject name

        Returns:
            List of schema versions with metadata
        """
        versions = self.client.get_subject_versions(subject)
        lineage = []

        for version in versions:
            url = f"{self.client.base_url}/subjects/{subject}/versions/{version}"
            response = requests.get(url, headers=self.client.headers)
            response.raise_for_status()

            data = response.json()
            lineage.append({
                'version': version,
                'id': data['id'],
                'schema': json.loads(data['schema']),
                'subject': data['subject']
            })

        return lineage
