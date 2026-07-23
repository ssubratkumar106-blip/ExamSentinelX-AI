"""
backend/app.py — Flask Application Factory
==========================================
PURPOSE: Create and configure the Flask application.

PATTERN: Application Factory
    Instead of creating `app = Flask(__name__)` globally,
    we use a function create_app() that builds and returns the app.

WHY?
    1. Allows different configs for testing vs production
    2. Prevents circular imports
    3. Enables multiple app instances (e.g., for testing)

WHAT IT DOES:
    1. Creates Flask app instance
    2. Loads configuration
    3. Initializes extensions (db, login, socketio)
    4. Registers blueprints (auth, exam, monitoring, admin)
    5. Registers SocketIO event handlers
    6. Sets up template context processors
"""

import os
from flask import Flask, render_template, redirect, url_for
from .config import get_config
from .extensions import db, login_manager, socketio, cors, mail, oauth


def create_app():
    """
    Application Factory Function.
    
    Returns:
        Flask app instance, fully configured and ready to run.
    """
    # ── Create Flask App ───────────────────────────────────────────────────────
    # template_folder points to our frontend/templates directory
    # static_folder points to our frontend/static directory
    app = Flask(
        __name__,
        template_folder=os.path.join(os.path.dirname(__file__), '..', 'frontend', 'templates'),
        static_folder=os.path.join(os.path.dirname(__file__), '..', 'frontend', 'static'),
        static_url_path='/static'
    )

    # ── Load Configuration ─────────────────────────────────────────────────────
    config_class = get_config()
    app.config.from_object(config_class)

    # ── Initialize Extensions ──────────────────────────────────────────────────
    db.init_app(app)
    login_manager.init_app(app)
    socketio.init_app(app, cors_allowed_origins="*")
    cors.init_app(app, resources={r"/api/*": {"origins": "*"}})
    mail.init_app(app)
    oauth.init_app(app)
    
    # Configure OAuth Providers
    if app.config.get('GOOGLE_CLIENT_ID'):
        oauth.register(
            name='google',
            client_id=app.config['GOOGLE_CLIENT_ID'],
            client_secret=app.config['GOOGLE_CLIENT_SECRET'],
            server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
            client_kwargs={'scope': 'openid email profile'}
        )
    

    @login_manager.user_loader
    def load_user(user_id):
        """Tell Flask-Login how to reload user from session cookie."""
        from database.models import User
        return User.query.get(int(user_id))

    # ── Register Blueprints ────────────────────────────────────────────────────
    # Blueprints are Flask's way to organize routes into modules
    from .auth.routes import auth_bp
    from .auth.oauth_routes import oauth_bp
    from .auth.password_routes import password_bp
    from .exam.routes import exam_bp
    from .monitoring.routes import monitoring_bp
    from .admin.routes import admin_bp

    app.register_blueprint(auth_bp, url_prefix='/auth')
    app.register_blueprint(oauth_bp, url_prefix='/auth')
    app.register_blueprint(password_bp, url_prefix='/auth')
    app.register_blueprint(exam_bp, url_prefix='/exam')
    app.register_blueprint(monitoring_bp, url_prefix='/monitor')
    app.register_blueprint(admin_bp, url_prefix='/admin')

    # ── Register SocketIO Events ───────────────────────────────────────────────
    from .monitoring.socket_events import register_socket_events
    register_socket_events(socketio)

    # ── Root Routes ────────────────────────────────────────────────────────────
    @app.route('/')
    def index():
        """Landing page — shows hero page for visitors, redirects dashboard for logged-in users."""
        from flask_login import current_user
        if current_user.is_authenticated:
            if current_user.is_admin:
                return redirect(url_for('admin.dashboard'))
            return redirect(url_for('exam.student_dashboard'))
        # Show the premium landing page for unauthenticated visitors
        return render_template('index.html')

    @app.route('/health')
    def health():
        """Health check endpoint for monitoring."""
        return {'status': 'ok', 'service': 'ExamSentinelX AI'}, 200

    # ── Error Handlers ─────────────────────────────────────────────────────────
    @app.errorhandler(404)
    def not_found(e):
        return render_template('errors/404.html'), 404

    @app.errorhandler(403)
    def forbidden(e):
        return render_template('errors/403.html'), 403

    @app.errorhandler(500)
    def server_error(e):
        return render_template('errors/500.html'), 500

    # ── Template Context Processors ────────────────────────────────────────────
    @app.context_processor
    def inject_globals():
        """Make these variables available in ALL templates."""
        from flask_login import current_user
        return {
            'app_name': 'ExamSentinelX AI',
            'current_user': current_user
        }

    return app
