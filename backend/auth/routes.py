"""
backend/auth/routes.py — Authentication Routes
================================================
PURPOSE: Handle user login, registration, and logout.

ROUTES:
    GET  /auth/login      → Show login form
    POST /auth/login      → Process login
    GET  /auth/register   → Show registration form
    POST /auth/register   → Process registration
    GET  /auth/logout     → Log out and redirect to login

SECURITY:
    - Passwords hashed with Werkzeug (bcrypt-based)
    - Flask-Login manages session cookies
    - Input validated before processing
"""

from flask import (Blueprint, render_template, request,
                   redirect, url_for, flash, jsonify)
from flask_login import login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

from backend.extensions import db
from database.models import User

# ── Create Blueprint ────────────────────────────────────────────────────────────
# A Blueprint groups related routes. url_prefix='/auth' is applied in app.py
auth_bp = Blueprint('auth', __name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LOGIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    """
    GET: Show the login page.
    POST: Validate credentials and start session.
    
    FLOW:
        1. Check if already logged in → redirect
        2. Get username/password from form
        3. Find user in database
        4. Verify password hash
        5. If valid: create session with Flask-Login
        6. Redirect to appropriate dashboard
    """
    # Already logged in → go to dashboard
    if current_user.is_authenticated:
        return _redirect_by_role(current_user)

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        remember = request.form.get('remember', False) == 'on'

        # ── Validate Input ─────────────────────────────────────────────────────
        if not username or not password:
            flash('Username and password are required.', 'error')
            return render_template('auth/login.html')

        # ── Find User ──────────────────────────────────────────────────────────
        user = User.query.filter_by(username=username).first()

        # ── Verify Password ────────────────────────────────────────────────────
        # SECURITY: We compare hashes, never plaintext passwords
        if not user or not check_password_hash(user.password_hash, password):
            flash('Invalid username or password. Please try again.', 'error')
            return render_template('auth/login.html', username=username)

        # ── Check Account Status ───────────────────────────────────────────────
        if not user.is_active:
            flash('Your account has been deactivated. Contact your administrator.', 'error')
            return render_template('auth/login.html')

        # ── Create Session ─────────────────────────────────────────────────────
        login_user(user, remember=remember)

        # Update last login time
        user.last_login = datetime.utcnow()
        db.session.commit()

        flash(f'Welcome back, {user.full_name}!', 'success')
        return _redirect_by_role(user)

    return render_template('auth/login.html')


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# REGISTER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    """
    GET: Show registration form.
    POST: Create new student account.
    
    NOTE: Only students can self-register.
    Admin accounts are created by seeding or by existing admins.
    """
    if current_user.is_authenticated:
        return _redirect_by_role(current_user)

    if request.method == 'POST':
        full_name = request.form.get('full_name', '').strip()
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')

        # ── Validation ─────────────────────────────────────────────────────────
        errors = _validate_registration(full_name, username, email, password, confirm_password)
        if errors:
            for error in errors:
                flash(error, 'error')
            return render_template('auth/register.html',
                                   full_name=full_name, username=username, email=email)

        # ── Check Uniqueness ───────────────────────────────────────────────────
        if User.query.filter_by(username=username).first():
            flash('Username already taken. Please choose another.', 'error')
            return render_template('auth/register.html', email=email)

        if User.query.filter_by(email=email).first():
            flash('Email already registered. Please log in instead.', 'error')
            return render_template('auth/register.html', username=username)

        # ── Create User ────────────────────────────────────────────────────────
        new_user = User(
            full_name=full_name,
            username=username,
            email=email,
            password_hash=generate_password_hash(password),
            role='student',
            is_active=True
        )
        db.session.add(new_user)
        db.session.commit()

        flash('Account created successfully! Please log in.', 'success')
        return redirect(url_for('auth.login'))

    return render_template('auth/register.html')


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LOGOUT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@auth_bp.route('/logout')
@login_required
def logout():
    """Log out the current user and clear session."""
    username = current_user.full_name
    logout_user()
    flash(f'Goodbye, {username}! Your session has ended.', 'info')
    return redirect(url_for('auth.login'))


# ── Helper Functions ────────────────────────────────────────────────────────────

def _redirect_by_role(user):
    """Redirect user to appropriate dashboard based on role."""
    if user.is_admin:
        return redirect(url_for('admin.dashboard'))
    return redirect(url_for('exam.student_dashboard'))


def _validate_registration(full_name, username, email, password, confirm_password):
    """
    Validate registration form inputs.
    Returns a list of error messages (empty if all valid).
    """
    errors = []

    if not full_name or len(full_name) < 2:
        errors.append('Full name must be at least 2 characters.')

    if not username or len(username) < 3:
        errors.append('Username must be at least 3 characters.')

    if not username.isalnum() and '_' not in username:
        errors.append('Username can only contain letters, numbers, and underscores.')

    if not email or '@' not in email:
        errors.append('Please enter a valid email address.')

    if not password or len(password) < 8:
        errors.append('Password must be at least 8 characters.')

    if password != confirm_password:
        errors.append('Passwords do not match.')

    return errors
