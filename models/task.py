from datetime import datetime
from models.database import db


class Task(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    title = db.Column(db.String(200), nullable=False)
    type = db.Column(db.String(50), nullable=False)
    subtype = db.Column(db.String(50))
    datetime = db.Column(db.DateTime, nullable=False)
    completed = db.Column(db.Boolean, default=False)
    duration = db.Column(db.Integer)  # in seconds
