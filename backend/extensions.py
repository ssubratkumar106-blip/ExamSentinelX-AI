"""
backend/extensions.py — Flask Extension Instances
==================================================
PURPOSE: Create extension instances SEPARATELY from the app.

WHY THIS PATTERN?
    Flask extensions need to be initialized with the app object.
    But if we create them inside app.py, we create circular imports.
    
    Solution: Create extensions here (no app attached yet),
    then call extension.init_app(app) inside create_app().
    
    This is called the "Application Factory Pattern" — a Flask best practice.

EXTENSIONS:
    - db: SQLAlchemy ORM for database operations
    - login_manager: Flask-Login for session management
    - socketio: Flask-SocketIO for real-time WebSocket communication
"""

from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from flask_socketio import SocketIO
from flask_cors import CORS
from flask_mail import Mail
from authlib.integrations.flask_client import OAuth

# ── Database ORM ───────────────────────────────────────────────────────────────
db = SQLAlchemy()

# ── Authentication ─────────────────────────────────────────────────────────────
login_manager = LoginManager()
login_manager.login_view = 'auth.login'           # Redirect here if not logged in
login_manager.login_message = 'Please log in to access ExamSentinelX.'
login_manager.login_message_category = 'warning'

# ── Real-time WebSocket ────────────────────────────────────────────────────────
# async_mode='threading' uses threading for WebSocket handling
socketio = SocketIO(cors_allowed_origins="*", async_mode='threading')

# ── CORS (Cross-Origin Resource Sharing) ──────────────────────────────────────
cors = CORS()

# ── Mail & OTP ─────────────────────────────────────────────────────────────────
mail = Mail()

# ── OAuth ──────────────────────────────────────────────────────────────────────
oauth = OAuth()
