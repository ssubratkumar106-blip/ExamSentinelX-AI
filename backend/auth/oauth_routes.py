from flask import Blueprint, url_for, redirect, flash
from flask_login import login_user, current_user
from backend.extensions import oauth, db
from database.models import User
from werkzeug.security import generate_password_hash
import uuid

oauth_bp = Blueprint('oauth_bp', __name__)

@oauth_bp.route('/login/google')
def login_google():
    redirect_uri = url_for('oauth_bp.auth_google', _external=True)
    return oauth.google.authorize_redirect(redirect_uri)

@oauth_bp.route('/callback/google')
def auth_google():
    try:
        token = oauth.google.authorize_access_token()
        user_info = oauth.google.parse_id_token(token, nonce=None)
        if not user_info:
            user_info = oauth.google.userinfo()
    except Exception as e:
        flash(f'Google authentication failed: {str(e)}', 'error')
        return redirect(url_for('auth.login'))

    email = user_info.get('email')
    full_name = user_info.get('name', '')
    google_id = str(user_info.get('sub', ''))

    if not email:
        flash('Google authentication failed: No email provided.', 'error')
        return redirect(url_for('auth.login'))

    user = User.query.filter_by(email=email).first()
    if user:
        if user.auth_provider != 'google':
            # Account linking strategy: Verify email ownership and upgrade/link
            user.auth_provider = 'google'
            user.oauth_id = google_id
            db.session.commit()
    else:
        # Create new user
        # We generate a random password hash since they authenticate via Google
        username = email.split('@')[0] + "_" + str(uuid.uuid4())[:4]
        user = User(
            full_name=full_name,
            username=username,
            email=email,
            password_hash=generate_password_hash(str(uuid.uuid4())),
            role='student',
            is_active=True,
            auth_provider='google',
            oauth_id=google_id
        )
        db.session.add(user)
        db.session.commit()

    login_user(user)
    flash(f'Welcome, {user.full_name}!', 'success')
    return redirect(url_for('exam.student_dashboard'))


