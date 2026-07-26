"""
backend/config.py — Configuration Management
=============================================
PURPOSE: Centralize all application configuration.

HOW IT WORKS:
    - Reads values from environment variables (loaded from .env)
    - Provides different configs for development, testing, production
    - Flask's app.config is populated from this class

CLEAN ARCHITECTURE PRINCIPLE:
    All config in one place → easy to change without touching other code.
"""

import os
from datetime import timedelta

# ── Project Root (absolute) ──────────────────────────────────────────────────
# This file is at: <project_root>/backend/config.py
# So going up two levels: backend/ → project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Config:
    """Base configuration — shared by all environments."""

    # ── Security ───────────────────────────────────────────────────────────────
    SECRET_KEY = os.getenv('SECRET_KEY', 'examsentinelx-dev-secret-key-2024-change-in-production')
    JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY', 'examsentinelx-jwt-key-2024')
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=8)  # 8 hours per exam session
    
    # ── OAuth ──────────────────────────────────────────────────────────────────
    GOOGLE_CLIENT_ID = os.getenv('GOOGLE_CLIENT_ID')
    GOOGLE_CLIENT_SECRET = os.getenv('GOOGLE_CLIENT_SECRET')
    
    # ── Mail / OTP ─────────────────────────────────────────────────────────────
    MAIL_SERVER = os.getenv('MAIL_SERVER', 'smtp.gmail.com')
    MAIL_PORT = int(os.getenv('MAIL_PORT', 587))
    MAIL_USE_TLS = os.getenv('MAIL_USE_TLS', 'True').lower() in ['true', '1', 't']
    MAIL_USERNAME = os.getenv('MAIL_USERNAME')
    MAIL_PASSWORD = os.getenv('MAIL_PASSWORD')
    MAIL_DEFAULT_SENDER = os.getenv('MAIL_DEFAULT_SENDER')

    # ── Database ───────────────────────────────────────────────────────────────
    # Use absolute path so SQLite works regardless of CWD
    _db_path = os.path.join(BASE_DIR, 'database', 'examsentinelx.db')
    
    _db_url = os.getenv('DATABASE_URL')
    if _db_url:
        # SQLAlchemy 1.4+ requires postgresql:// instead of postgres://
        if _db_url.startswith("postgres://"):
            _db_url = _db_url.replace("postgres://", "postgresql://", 1)
        
        # Supabase and many cloud providers require SSL
        if "supabase" in _db_url or "pooler" in _db_url or "render" in _db_url:
            if "?" not in _db_url:
                _db_url += "?sslmode=require"
            elif "sslmode=" not in _db_url:
                _db_url += "&sslmode=require"
                
    SQLALCHEMY_DATABASE_URI = _db_url or f'sqlite:///{_db_path}'
    SQLALCHEMY_TRACK_MODIFICATIONS = False  # Disable to save memory

    # ── File Storage ───────────────────────────────────────────────────────────
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'captures', 'evidence')
    REPORTS_FOLDER = os.path.join(BASE_DIR, 'reports', 'generated')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max upload

    # ── SocketIO ───────────────────────────────────────────────────────────────
    SOCKETIO_ASYNC_MODE = 'threading'  # Use threading instead of deprecated eventlet

    # ── AI Configuration ──────────────────────────────────────────────────────
    YOLO_MODEL_PATH = os.getenv('YOLO_MODEL_PATH', os.path.join(BASE_DIR, 'ai', 'models', 'yolov8n.pt'))
    CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', '0.5'))
    FACE_ABSENCE_THRESHOLD_SEC = int(os.getenv('FACE_ABSENCE_THRESHOLD_SEC', '3'))
    HEAD_POSE_YAW_LIMIT = int(os.getenv('HEAD_POSE_YAW_LIMIT', '30'))
    HEAD_POSE_PITCH_LIMIT = int(os.getenv('HEAD_POSE_PITCH_LIMIT', '20'))


class DevelopmentConfig(Config):
    """Development environment — verbose debugging enabled."""
    DEBUG = True
    TESTING = False


class TestingConfig(Config):
    """Testing environment — uses in-memory SQLite."""
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
    WTF_CSRF_ENABLED = False


class ProductionConfig(Config):
    """Production environment — security hardened."""
    DEBUG = False
    TESTING = False


# ── Config selector ────────────────────────────────────────────────────────────
config_map = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
}

def get_config():
    """Return the appropriate config class based on FLASK_ENV."""
    env = os.getenv('FLASK_ENV', 'development')
    return config_map.get(env, DevelopmentConfig)
