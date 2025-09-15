"""
Main entry point for the API Gateway FastAPI application.

This module initializes and configures the core components of the API Gateway,
including:
- Logging setup.
- Application lifecycle management (startup and shutdown).
- Middleware for request logging, rate limiting, and authentication.
- Router inclusion for internal and proxy endpoints.
- Monitoring and metrics setup.
"""

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from contextlib import asynccontextmanager
from src.api import health, metrics
from src.api import ticket_ml
from src.core import monitoring, routing, auth
from src.core.rate_limit import RateLimitMiddleware
from src.core.logging import setup_logging, RequestLoggingMiddleware, get_logger
from typing import AsyncGenerator


# Set up logging as the very first step to ensure all subsequent events are logged.
setup_logging()
logger = get_logger("main")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Manages the lifespan of the FastAPI application.

    This context manager handles startup and shutdown logic, ensuring resources
    are properly initialized before the server starts and cleaned up when it
    shuts down.

    Args:
        app: The FastAPI application instance.

    Yields:
        None: The control is yielded to the application's runtime.
    """
    logger.info("Starting up API Gateway", extra={"extra_fields": {"service_version": "1.0.0"}})
    yield
    logger.info("Shutting down API Gateway")


app = FastAPI(title="API Gateway", lifespan=lifespan)

# ---- Middleware Chain ----
# Middleware is added in the order of execution for incoming requests.
# It is processed in reverse order for outgoing responses.

# 1. Request logging middleware: Applied first to capture and log every request and its metadata.
app.add_middleware(RequestLoggingMiddleware)

# 2. Rate limiting middleware: Protects endpoints from excessive requests.
app.add_middleware(RateLimitMiddleware)

# 3. JWT Auth middleware: Authenticates requests and handles JWT token validation.
app.add_middleware(auth.AuthMiddleware)


# ---- Router Inclusion ----
# Register API endpoints from different modules.

# Register internal-facing endpoints for health checks and metrics.
app.include_router(health.router, prefix="/health", tags=["internal"])
app.include_router(metrics.router, prefix="/metrics", tags=["internal"])

# Register the primary routing/proxy endpoints.
app.include_router(routing.router, prefix="/api", tags=["proxy"])

app.include_router(ticket_ml.router, prefix="/ml", tags=["ml"])
# ---- Monitoring Setup ----
# Configure monitoring and metrics collection for the application.
monitoring.setup_metrics(app)

# Log startup (outside lifespan, runs on import)
logger.info("API Gateway started successfully")

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title=app.title,
        version="1.0.0",
        description="API Gateway with JWT Authentication",
        routes=app.routes,
    )
    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT"
        }
    }
    # Apply BearerAuth as default for all endpoints
    openapi_schema["security"] = [{"BearerAuth": []}]
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi