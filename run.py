"""
ExamSentinelX AI — Application Entry Point
=======================================
This is the main entry point for the ExamSentinelX AI application.

How to run:
    python run.py

What it does:
    1. Loads environment variables from .env file
    2. Creates the Flask application using the factory pattern
    3. Initializes the database tables
    4. Starts the development server with SocketIO support
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file BEFORE importing anything else
# This ensures all config values are available during app creation
load_dotenv()

# Limit CPU threads for heavy AI libraries to prevent OOM on Render Free Tier
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from backend.app import create_app
from backend.extensions import socketio, db

# ── Create Flask Application ──────────────────────────────────────────────────
app = create_app()

# ── Initialize Database ────────────────────────────────────────────────────────
with app.app_context():
    try:
        db.create_all()
        
        # Seed admin user and sample data if DB is empty
        from database.seed import seed_database
        seed_database()
        print("Database initialized successfully.")
    except Exception as e:
        print(f"⚠️ Warning: Could not initialize database during startup.")
        print(f"⚠️ Error details: {e}")
        print("⚠️ The application will continue to boot, but database features may fail until the connection is restored.")

# ── Start Server ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'
    
    db_uri = app.config.get('SQLALCHEMY_DATABASE_URI', '')
    db_type = 'Supabase / PostgreSQL (Cloud)' if db_uri.startswith('postgres') else 'SQLite (Local)'
    
    print("=" * 60)
    print("  ExamSentinelX AI -- Starting Server")
    print("=" * 60)
    print(f"  URL:   http://localhost:{port}")
    print(f"  DB:    {db_type}")
    print(f"  Debug: {debug}")
    print(f"  Admin: http://localhost:{port}/admin/dashboard")
    print("=" * 60)
    
    # Use socketio.run instead of app.run to enable WebSocket support
    socketio.run(app, host=host, port=port, debug=debug,
                 use_reloader=False, allow_unsafe_werkzeug=True)
