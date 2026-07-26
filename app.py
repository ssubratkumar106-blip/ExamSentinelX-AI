import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from backend.app import create_app
from backend.extensions import socketio, db

# ── Create Flask Application ──────────────────────────────────────────────────
app = create_app()

# ── Initialize Database ────────────────────────────────────────────────────────
with app.app_context():
    db.create_all()
    
    # Seed admin user and sample data if DB is empty
    from database.seed import seed_database
    seed_database()

# ── Start Server ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # Hugging Face Spaces expects the app to run on port 7860
    host = '0.0.0.0'
    port = 7860
    debug = False
    
    db_uri = app.config.get('SQLALCHEMY_DATABASE_URI', '')
    db_type = 'Supabase / PostgreSQL (Cloud)' if db_uri.startswith('postgres') else 'SQLite (Local)'
    
    print("=" * 60)
    print("  ExamSentinelX AI -- Starting Server on Hugging Face")
    print("=" * 60)
    print(f"  URL:   http://localhost:{port}")
    print(f"  DB:    {db_type}")
    print("=" * 60)
    
    # Use socketio.run instead of app.run to enable WebSocket support
    socketio.run(app, host=host, port=port, debug=debug,
                 use_reloader=False, allow_unsafe_werkzeug=True)
