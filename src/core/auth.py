import os
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from jose import jwt, JWTError, ExpiredSignatureError
from datetime import datetime, timedelta, timezone
from src.core.logging import LoggerMixin

# JWT configuration
SECRET_KEY = os.getenv("JWT_SECRET", "supersecretjwtkey")
ALGORITHM = "HS256"


class AuthMiddleware(BaseHTTPMiddleware, LoggerMixin):
   """
   JWT authentication middleware for protecting API routes.
   
   Validates Bearer tokens in the Authorization header and attaches
   user information to request state. Skips authentication for
   internal endpoints like /health and /metrics.
   """

   async def dispatch(self, request: Request, call_next):
       """
       Process authentication for incoming requests.
       
       Args:
           request: Incoming HTTP request
           call_next: Next middleware in chain
           
       Returns:
           HTTP response or 401/500 for authentication failures
       """
       request_id = getattr(request.state, "request_id", "unknown")
       
              # Skip authentication for internal + docs endpoints
       if (
           request.url.path.startswith("/health")
           or request.url.path.startswith("/metrics")
           or request.url.path.startswith("/docs")
           or request.url.path.startswith("/redoc")
           or request.url.path.startswith("/openapi.json")
       ):
           self.log_info(
               "Skipping auth for internal/docs endpoint",
               request_id=request_id,
               path=request.url.path
           )
           return await call_next(request)


       # Check for Authorization header
       auth_header = request.headers.get("Authorization")
       if not auth_header:
           self.log_warning(
               "Missing authorization header",
               request_id=request_id,
               path=request.url.path,
               client_ip=request.client.host
           )
           return JSONResponse(
               status_code=401,
               content={"detail": "Authorization header missing"}
           )

       # Parse Bearer token
       parts = auth_header.split()
       if len(parts) != 2 or parts[0].lower() != "bearer":
           self.log_warning(
               "Invalid authorization header format",
               request_id=request_id,
               path=request.url.path,
               auth_header_parts=len(parts)
           )
           return JSONResponse(
               status_code=401,
               content={"detail": "Invalid Authorization header format"}
           )

       token = parts[1]

       try:
           # Decode and validate JWT token
           payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
           request.state.user = payload
           
           user_id = payload.get("sub", "unknown")
           self.log_info(
               "Authentication successful",
               request_id=request_id,
               user_id=user_id,
               path=request.url.path
           )
           
       except ExpiredSignatureError:
           self.log_warning(
               "Expired JWT token",
               request_id=request_id,
               path=request.url.path
           )
           return JSONResponse(
               status_code=401,
               content={"detail": "Token expired"}
           )
       except JWTError as e:
           self.log_warning(
               "Invalid JWT token",
               request_id=request_id,
               path=request.url.path,
               jwt_error=str(e)
           )
           return JSONResponse(
               status_code=401,
               content={"detail": f"Invalid token: {str(e)}"}
           )
       except Exception as e:
           self.log_error(
               "Unexpected error during authentication",
               error=e,
               request_id=request_id,
               path=request.url.path
           )
           return JSONResponse(
               status_code=500,
               content={"detail": f"Unexpected error: {str(e)}"}
           )

       return await call_next(request)


def create_jwt_token(data: dict, expires_in: int = 3600) -> str:
   """
   Generate a JWT token with expiration.
   
   Args:
       data: Payload data to include in token
       expires_in: Token expiration time in seconds (default: 1 hour)
       
   Returns:
       Encoded JWT token string
   """
   expire = datetime.now(timezone.utc) + timedelta(seconds=expires_in)
   payload = {**data, "exp": expire}
   return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)