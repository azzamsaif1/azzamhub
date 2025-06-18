from models.database import db


class User(db.Model):
    __tablename__ = 'users'  # ✅ هذا يحل الخطأ

    id = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

    tasks = db.relationship('Task', backref='user', lazy=True)
