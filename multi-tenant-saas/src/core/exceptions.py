"""
Core exception classes for multi-tenant SaaS platform.

Exception design philosophy:
1. Every exception includes business impact context
2. Exceptions are logged with cost implications
3. Recovery strategies are documented

Based on patterns from:
- Stripe's error codes (card_declined, authentication_failed, etc.)
- AWS error taxonomy (throttling, permissions, resources)
- Google Cloud's structured errors
"""

from typing import Optional, Dict, Any


class SaaSException(Exception):
    """
    Base exception for all platform errors.

    Every exception includes:
    - Error code (for client handling)
    - User message (safe to show to users)
    - Internal message (for debugging)
    - HTTP status code (for API responses)
    - Cost impact (for monitoring)
    """

    def __init__(
        self,
        message: str,
        error_code: str,
        user_message: Optional[str] = None,
        http_status: int = 500,
        cost_impact: float = 0.0,
        recovery_hint: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.user_message = user_message or "An error occurred. Please try again."
        self.http_status = http_status
        self.cost_impact = cost_impact
        self.recovery_hint = recovery_hint
        self.metadata = kwargs

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for API responses"""
        return {
            "error": {
                "code": self.error_code,
                "message": self.user_message,
                "type": self.__class__.__name__
            }
        }


# ============================================================================
# TENANT ISOLATION ERRORS
# ============================================================================

class TenantIsolationError(SaaSException):
    """
    Critical: Tenant isolation violation.

    Business impact: Potential data breach ($150K - $5M+)
    Response: Immediate alert to security team
    """

    def __init__(self, message: str, tenant_id: str, **kwargs):
        super().__init__(
            message=message,
            error_code="tenant_isolation_violation",
            user_message="Access denied for security reasons.",
            http_status=403,
            cost_impact=500_000.0,  # Conservative breach estimate
            recovery_hint="Check tenant_context and RLS policies",
            tenant_id=tenant_id,
            **kwargs
        )


class TenantContextMissingError(SaaSException):
    """
    Missing tenant context in request.

    This should NEVER happen in production—it means a code path bypassed
    our tenant middleware.

    Business impact: High risk of cross-tenant access
    """

    def __init__(self, message: str = "Tenant context not set", **kwargs):
        super().__init__(
            message=message,
            error_code="tenant_context_missing",
            user_message="Invalid request. Please try again.",
            http_status=400,
            cost_impact=10_000.0,  # Incident investigation cost
            recovery_hint="Ensure tenant_middleware is applied to all routes",
            **kwargs
        )


class TenantNotFoundError(SaaSException):
    """Tenant doesn't exist in system"""

    def __init__(self, tenant_id: str, **kwargs):
        super().__init__(
            message=f"Tenant not found: {tenant_id}",
            error_code="tenant_not_found",
            user_message="Account not found.",
            http_status=404,
            cost_impact=50.0,  # Support ticket cost
            tenant_id=tenant_id,
            **kwargs
        )


# ============================================================================
# RESOURCE LIMIT ERRORS
# ============================================================================

class ResourceLimitExceededError(SaaSException):
    """
    Customer exceeded tier limits.

    Business impact: Upsell opportunity!
    Response: Show upgrade CTA
    """

    def __init__(
        self,
        resource_type: str,
        limit: int,
        usage: int,
        tenant_id: str,
        **kwargs
    ):
        super().__init__(
            message=f"Resource limit exceeded: {resource_type} (limit: {limit}, usage: {usage})",
            error_code="resource_limit_exceeded",
            user_message=f"You've reached your {resource_type} limit. Upgrade for more capacity.",
            http_status=429,  # Too Many Requests
            cost_impact=-200.0,  # Negative = revenue opportunity!
            recovery_hint="Offer tier upgrade or overage pricing",
            resource_type=resource_type,
            limit=limit,
            usage=usage,
            tenant_id=tenant_id,
            **kwargs
        )


class RateLimitExceededError(SaaSException):
    """
    API rate limit exceeded.

    Business impact: Prevents abuse, maintains SLA for others
    """

    def __init__(
        self,
        tenant_id: str,
        limit: int,
        window_seconds: int,
        retry_after_seconds: int,
        **kwargs
    ):
        super().__init__(
            message=f"Rate limit exceeded: {limit} requests per {window_seconds}s",
            error_code="rate_limit_exceeded",
            user_message=f"Too many requests. Please try again in {retry_after_seconds} seconds.",
            http_status=429,
            cost_impact=0.0,  # Prevented cost!
            recovery_hint="Implement exponential backoff in client",
            tenant_id=tenant_id,
            retry_after_seconds=retry_after_seconds,
            **kwargs
        )


# ============================================================================
# DATABASE ERRORS
# ============================================================================

class DatabaseConnectionError(SaaSException):
    """
    Can't connect to database.

    Business impact: $9K/minute downtime (enterprise)
    Response: Circuit breaker should prevent cascading failures
    """

    def __init__(self, message: str, pool_name: Optional[str] = None, **kwargs):
        super().__init__(
            message=message,
            error_code="database_connection_failed",
            user_message="Service temporarily unavailable. Please try again.",
            http_status=503,
            cost_impact=9_000.0,  # Per-minute cost for enterprise
            recovery_hint="Check connection pool health and circuit breaker state",
            pool_name=pool_name,
            **kwargs
        )


class ConnectionPoolExhaustedError(SaaSException):
    """
    No available database connections.

    Business impact: Service degradation
    Root cause: Usually noisy neighbor or sudden traffic spike
    """

    def __init__(self, pool_name: str, max_connections: int, **kwargs):
        super().__init__(
            message=f"Connection pool exhausted: {pool_name} (max: {max_connections})",
            error_code="connection_pool_exhausted",
            user_message="Service is experiencing high load. Please try again shortly.",
            http_status=503,
            cost_impact=500.0,  # Per-minute impact
            recovery_hint="Check for noisy neighbors or scale connection pool",
            pool_name=pool_name,
            max_connections=max_connections,
            **kwargs
        )


# ============================================================================
# BILLING ERRORS
# ============================================================================

class PaymentRequiredError(SaaSException):
    """
    Account has unpaid invoices.

    Business impact: Churn risk if handled poorly
    Response: Grace period, then read-only access
    """

    def __init__(self, tenant_id: str, outstanding_amount: float, **kwargs):
        super().__init__(
            message=f"Payment required: ${outstanding_amount:.2f} outstanding",
            error_code="payment_required",
            user_message=f"Your account has an outstanding balance of ${outstanding_amount:.2f}. Please update payment method.",
            http_status=402,  # Payment Required
            cost_impact=outstanding_amount,
            recovery_hint="Show payment update CTA",
            tenant_id=tenant_id,
            outstanding_amount=outstanding_amount,
            **kwargs
        )


class TierDowngradeError(SaaSException):
    """
    Can't downgrade due to usage.

    Business impact: Prevents revenue loss
    """

    def __init__(self, current_tier: str, target_tier: str, reason: str, **kwargs):
        super().__init__(
            message=f"Cannot downgrade from {current_tier} to {target_tier}: {reason}",
            error_code="tier_downgrade_blocked",
            user_message=f"Cannot downgrade: {reason}",
            http_status=400,
            cost_impact=-100.0,  # Prevented revenue loss
            recovery_hint="Show usage reduction steps",
            current_tier=current_tier,
            target_tier=target_tier,
            **kwargs
        )


# ============================================================================
# ML MODEL ERRORS
# ============================================================================

class ModelNotFoundError(SaaSException):
    """Requested ML model doesn't exist"""

    def __init__(self, model_id: str, tenant_id: str, **kwargs):
        super().__init__(
            message=f"Model not found: {model_id}",
            error_code="model_not_found",
            user_message="The requested model was not found.",
            http_status=404,
            cost_impact=50.0,
            model_id=model_id,
            tenant_id=tenant_id,
            **kwargs
        )


class ModelInferenceError(SaaSException):
    """Error during model prediction"""

    def __init__(self, model_id: str, error_detail: str, **kwargs):
        super().__init__(
            message=f"Model inference failed: {error_detail}",
            error_code="model_inference_failed",
            user_message="Prediction failed. Please check your input data.",
            http_status=500,
            cost_impact=100.0,  # Investigation cost
            recovery_hint="Check model health and input validation",
            model_id=model_id,
            **kwargs
        )


# ============================================================================
# PERMISSION ERRORS
# ============================================================================

class FeatureNotAvailableError(SaaSException):
    """
    Feature not available in current tier.

    Business impact: Upsell opportunity!
    """

    def __init__(
        self,
        feature_name: str,
        current_tier: str,
        required_tier: str,
        **kwargs
    ):
        super().__init__(
            message=f"Feature '{feature_name}' requires {required_tier} tier",
            error_code="feature_not_available",
            user_message=f"This feature requires the {required_tier} plan. Upgrade to access it!",
            http_status=403,
            cost_impact=-300.0,  # Upsell opportunity
            recovery_hint="Show tier comparison and upgrade CTA",
            feature_name=feature_name,
            current_tier=current_tier,
            required_tier=required_tier,
            **kwargs
        )


# ============================================================================
# VALIDATION ERRORS
# ============================================================================

class InvalidInputError(SaaSException):
    """Invalid user input"""

    def __init__(self, field: str, reason: str, **kwargs):
        super().__init__(
            message=f"Invalid input for field '{field}': {reason}",
            error_code="invalid_input",
            user_message=f"Invalid {field}: {reason}",
            http_status=400,
            cost_impact=0.0,
            field=field,
            **kwargs
        )
