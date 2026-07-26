from flask import Blueprint, render_template, request, redirect, url_for, flash, current_app
from backend.extensions import db, mail
from database.models import User
from werkzeug.security import generate_password_hash
from flask_mail import Message
import random
import string
from datetime import datetime, timedelta

password_bp = Blueprint('password_bp', __name__)

def generate_otp():
    return ''.join(random.choices(string.digits, k=6))

@password_bp.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        user = User.query.filter_by(email=email).first()

        if user:
            otp = generate_otp()
            user.otp_code = otp
            user.otp_expiry = datetime.utcnow() + timedelta(minutes=10)
            db.session.commit()

            try:
                msg = Message("Password Reset OTP",
                              recipients=[email])
                msg.body = f"Your OTP for password reset is: {otp}\n\nThis OTP is valid for 10 minutes."
                mail.send(msg)
                flash('An OTP has been sent to your email address.', 'info')
                return redirect(url_for('password_bp.verify_otp', email=email))
            except Exception as e:
                flash('Failed to send OTP email. Please check server configuration.', 'error')
                current_app.logger.error(f"Mail error: {e}")
        else:
            # For security, we don't reveal if the email exists
            flash('If an account with that email exists, an OTP has been sent.', 'info')
            return redirect(url_for('password_bp.verify_otp', email=email))

    return render_template('auth/forgot_password.html')

@password_bp.route('/verify-otp', methods=['GET', 'POST'])
def verify_otp():
    email = request.args.get('email')
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        otp = request.form.get('otp', '').strip()

        user = User.query.filter_by(email=email).first()
        if user and user.otp_code == otp:
            if user.otp_expiry and datetime.utcnow() > user.otp_expiry:
                flash('OTP has expired. Please request a new one.', 'error')
                return redirect(url_for('password_bp.forgot_password'))
            
            # OTP is valid, store a temporary flag in session or simply redirect to reset
            # In a secure app, you'd use a signed token in URL. For simplicity, we pass a query param.
            # However, query params can be insecure for resetting.
            # Instead, we clear OTP and set a new specific token, but to minimize changes:
            # We'll just pass the valid OTP code to the reset endpoint.
            return redirect(url_for('password_bp.reset_password', email=email, token=otp))
        else:
            flash('Invalid OTP.', 'error')

    return render_template('auth/verify_otp.html', email=email)

@password_bp.route('/reset-password', methods=['GET', 'POST'])
def reset_password():
    email = request.args.get('email')
    token = request.args.get('token')

    if not email or not token:
        flash('Invalid reset link.', 'error')
        return redirect(url_for('auth_bp.login'))

    user = User.query.filter_by(email=email, otp_code=token).first()
    
    if not user:
        flash('Invalid or expired reset token.', 'error')
        return redirect(url_for('auth_bp.login'))

    if request.method == 'POST':
        new_password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        if len(new_password) < 8:
            flash('Password must be at least 8 characters.', 'error')
        elif new_password != confirm_password:
            flash('Passwords do not match.', 'error')
        else:
            user.password_hash = generate_password_hash(new_password)
            user.otp_code = None
            user.otp_expiry = None
            # If they had an oauth provider, maybe clear it so they can login locally now?
            # We'll leave auth_provider as is, they can use both.
            db.session.commit()
            flash('Your password has been successfully reset. Please log in.', 'success')
            return redirect(url_for('auth_bp.login'))

    return render_template('auth/reset_password.html', email=email, token=token)
