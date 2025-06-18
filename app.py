from flask import Flask, render_template, request, redirect, url_for, session, flash, g,jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from models.database import db
from models.user import User
from models.task import Task
from datetime import datetime,timedelta
from models.session import Session
from models.invest import FinanceEntry
from flask_cors import CORS
import openai
import os
from duckduckgo_search import DDGS
from keybert import KeyBERT
from deep_translator import GoogleTranslator
from keybert import KeyBERT
from transformers import pipeline
import random
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

import yfinance as yf
from sklearn.linear_model import LinearRegression
import numpy as np


from transformers import pipeline
from pyresparser import ResumeParser
import pdfplumber
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import spacy
from translate import Translator
nltk.download('stopwords')




app = Flask(__name__)
app.secret_key = "azzamhub_secret"
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

# ========== إنشاء قاعدة البيانات ==========
with app.app_context():
    db.create_all()
# ========== Helper Functions ==========
def get_productivity_summary(user_id):
    today = datetime.now().date()
    week_ago = today - timedelta(days=7)
    
    # Tasks stats
    total_tasks = Task.query.filter_by(user_id=user_id).count()
    completed_tasks = Task.query.filter_by(user_id=user_id, completed=True).count()
    
    # Time stats
    total_focus_time = db.session.query(db.func.sum(Session.duration)).filter(
        Session.user_id == user_id,
        db.func.date(Session.start_time) == today
    ).scalar() or 0
    
    weekly_focus_time = db.session.query(db.func.sum(Session.duration)).filter(
        Session.user_id == user_id,
        Session.start_time >= week_ago
    ).scalar() or 0
    
    return {
        'total_tasks': total_tasks,
        'completed_tasks': completed_tasks,
        'today_focus_time': total_focus_time,
        'weekly_focus_time': weekly_focus_time,
        'completion_rate': (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0
    }




# app.py (أعلى الملف)

# ========== Routes ==========
@app.before_request
def load_user():
    g.user = None
    if "user_id" in session:
        g.user = User.query.get(session["user_id"])

@app.route("/")
def home():
    return redirect(url_for("dashboard"))

@app.route("/auth", methods=["GET"])
def auth_page():
    form_type = request.args.get("form", "login")
    return render_template("auth.html", form_type=form_type)

@app.route("/register", methods=["POST"])
def register():
    name = request.form.get("name")
    email = request.form.get("email")
    password = request.form.get("password")

    if User.query.filter_by(email=email).first():
        flash("Email already registered.", "error")
        return redirect(url_for("auth_page", form="register"))

    hashed_pw = generate_password_hash(password)
    new_user = User(full_name=name, email=email, password=hashed_pw)
    db.session.add(new_user)
    db.session.commit()

    flash("Account created successfully! Please log in.", "success")
    return redirect(url_for("auth_page", form="login"))

@app.route("/login", methods=["POST"])
def login():
    email = request.form.get("email")
    password = request.form.get("password")

    user = User.query.filter_by(email=email).first()
    if user and check_password_hash(user.password, password):
        session["user_id"] = user.id
        session["user_name"] = user.full_name
        flash("Login successful!", "success")
        return redirect("/dashboard")
    else:
        flash("Invalid credentials", "error")
        return redirect(url_for("auth_page", form="login"))

@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out", "info")
    return redirect(url_for("auth_page", form="login"))

@app.route("/dashboard")
def dashboard():
    if not g.user:
        flash("Please log in to access your dashboard.", "error")
        return redirect(url_for("auth_page", form_type="login"))

    summary = get_productivity_summary(g.user.id)
    tasks = Task.query.filter_by(user_id=g.user.id).order_by(Task.datetime.desc()).all()
    return render_template("dashboard.html", user=g.user, tasks=tasks, summary=summary)

@app.route("/planner")
def planner():
    if not g.user:
        return redirect(url_for("auth_page", form="login"))
    
    tasks = Task.query.filter_by(user_id=g.user.id).order_by(Task.datetime.desc()).all()
    summary = get_productivity_summary(g.user.id)
    return render_template("planner.html", tasks=tasks, summary=summary)

@app.route("/add_task", methods=["POST"])
def add_task():
    if not g.user:
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json()
    title = data.get("title")
    task_type = data.get("type")
    subtype = data.get("subtype")
    datetime_str = data.get("datetime")

    if not all([title, task_type, datetime_str]):
        return jsonify({"error": "Missing data"}), 400

    try:
        datetime_obj = datetime.strptime(datetime_str, "%Y-%m-%dT%H:%M")
    except ValueError:
        return jsonify({"error": "Invalid datetime format"}), 400

    new_task = Task(
        user_id=g.user.id,
        title=title,
        type=task_type,
        subtype=subtype,
        datetime=datetime_obj
    )

    db.session.add(new_task)
    db.session.commit()
    return jsonify({"message": "Task added", "task_id": new_task.id}), 200

@app.route("/get_tasks", methods=["GET"])
def get_tasks():
    if not g.user:
        return jsonify({"error": "Unauthorized"}), 401

    tasks = Task.query.filter_by(user_id=g.user.id).all()
    task_list = []
    for task in tasks:
        task_list.append({
            "id": task.id,
            "title": task.title,
            "type": task.type,
            "subtype": task.subtype,
            "datetime": task.datetime.strftime("%Y-%m-%dT%H:%M"),
            "completed": task.completed,
            "createdAt": task.datetime.isoformat()
        })
    return jsonify(task_list)

@app.route("/start_timer", methods=["POST"])
def start_timer():
    if not g.user:
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json()
    task_id = data.get("task_id")
    
    task = Task.query.get(task_id)
    if not task or task.user_id != g.user.id:
        return jsonify({"error": "Task not found"}), 404

    new_session = Session(
        user_id=g.user.id,
        task_id=task.id,
        start_time=datetime.now(),
        duration=0
    )
    db.session.add(new_session)
    db.session.commit()
    
    return jsonify({
        "message": "Timer started",
        "session_id": new_session.id
    }), 200

@app.route("/stop_timer", methods=["POST"])
def stop_timer():
    if not g.user:
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json()
    session_id = data.get("session_id")
    
    session = Session.query.get(session_id)
    if not session or session.user_id != g.user.id:
        return jsonify({"error": "Session not found"}), 404

    end_time = datetime.now()
    duration = (end_time - session.start_time).total_seconds()
    
    session.end_time = end_time
    session.duration = duration
    db.session.commit()
    
    # Update task duration if needed
    task = Task.query.get(session.task_id)
    if task:
        task.duration = (task.duration or 0) + duration
        db.session.commit()
    
    return jsonify({
        "message": "Timer stopped",
        "duration": duration
    }), 200

@app.route("/get_summary", methods=["GET"])
def get_summary():
    if not g.user:
        return jsonify({"error": "Unauthorized"}), 401

    summary = get_productivity_summary(g.user.id)
    return jsonify(summary)

@app.route("/get_ai_suggestions")
def get_ai_suggestions():
    if not g.user:
        return jsonify({"suggestions": ["🔒 يرجى تسجيل الدخول لعرض الاقتراحات"]})
    
    # يمكن تحسين هذا الذكاء لاحقًا بناءً على مهام المستخدم
    suggestions = []

    # إذا لم يكمل أي مهمة مؤخراً، نقترح مهام بسيطة
    last_task = Task.query.filter_by(user_id=g.user.id).order_by(Task.datetime.desc()).first()
    if not last_task:
        suggestions = [
            "ابدأ اليوم بتحديد هدف بسيط",
            "جرب كتابة خطة ليومك في ورقة",
            "ابدأ بمهمة خفيفة تحبها"
        ]
    else:
        suggestions = [
            f"راجع المهمة الأخيرة: {last_task.title}",
            "حدد مهمة تكمل ما بدأت سابقًا",
            "خذ استراحة قصيرة قبل المهمة التالية",
            "اقترح تطوير {last_task.type} إلى مستوى متقدم"
        ]
    
    return jsonify({"suggestions": suggestions})

#cv
@app.route("/cv")
def cv_page():
    if not g.user:
        flash("Please log in to access your CV.", "error")
        return redirect(url_for("auth_page", form="login"))
    return render_template("cv.html", user=g.user)
#enhance
# تهيئة النموذج مرة واحدة فقط
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

@app.route("/enhance_summary", methods=["POST"])
def enhance_summary():
    data = request.get_json()
    text = data.get("text", "")
    if not text.strip():
        return jsonify({"error": "No text provided"}), 400

    try:
        input_length = len(text.split())
        max_len = min(max(input_length - 5, 20), 130)  # بين 20 و130
        min_len = max(10, int(max_len / 2))            # نصف max_len

        result = summarizer(text, max_length=max_len, min_length=min_len, do_sample=False)
        return jsonify({"enhanced": result[0]["summary_text"]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
    


# ✅ صياغة فقرة ذكية عامة لأي كلمات
def build_paragraph(words):
    if not words:
        return ""

    capitalized = [word.capitalize() for word in words if word.isalpha()]
    listed_skills = ", ".join(capitalized[:-1]) + " und " + capitalized[-1] if len(capitalized) > 1 else capitalized[0]

    paragraph = (
        f"Ich habe mich kontinuierlich mit Schlüsselthemen wie {listed_skills} auseinandergesetzt. "
        "Diese Kompetenzen möchte ich in meiner zukünftigen beruflichen Entwicklung weiter vertiefen und anwenden."
    )
    return paragraph

@app.route("/ats", methods=["POST"])
def ats_optimize():
    if "resume" not in request.files or "job_desc" not in request.form:
        return jsonify({"error": "Resume or job description missing"}), 400

    resume_file = request.files["resume"]
    job_desc = request.form["job_desc"]

    resume_path = os.path.join("uploads", resume_file.filename)
    os.makedirs("uploads", exist_ok=True)
    resume_file.save(resume_path)

    try:
        parsed = ResumeParser(resume_path).get_extracted_data()
        resume_text = " ".join(parsed.get("skills", []))
    except:
        with pdfplumber.open(resume_path) as pdf:
            resume_text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
        parsed = {"skills": []}

    texts = [resume_text, job_desc]
    vectorizer = CountVectorizer().fit_transform(texts)
    similarity = cosine_similarity(vectorizer)[0][1] * 100

    job_keywords = job_desc.lower().split()
    resume_keywords = resume_text.lower().split()
    missing_skills = [word for word in job_keywords if word not in resume_keywords]
    common_words = {"and", "or", "the", "with", "a", "an", "in", "to", "of", "for"}
    recommended_skills = [word for word in missing_skills if word not in common_words and len(word) > 2]
    try:
        parsed = ResumeParser(resume_path).get_extracted_data()
        resume_text = " ".join(parsed.get("skills", []))
        # ✅ تحقق إضافي: إذا لم يُستخرج شيء، استخدم pdfplumber
        if not resume_text.strip():
            raise ValueError("Empty skills")
    except:
        with pdfplumber.open(resume_path) as pdf:
            resume_text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
        parsed = {"skills": []}

    generated_paragraph = build_paragraph(recommended_skills)



    return jsonify({
        "match": round(similarity, 2),
        "resume_skills": parsed.get("skills", []),
        "raw_resume": resume_text,
        
        "suggested_skills": recommended_skills,
        "generated_paragraph": generated_paragraph  # ✅    
    })



nlp = spacy.load("de_core_news_md")
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

def extract_resume_data(path):
    try:
        parsed = ResumeParser(path, custom_nlp=nlp).get_extracted_data()
        if not parsed or not parsed.get("skills"):
            raise ValueError("No skills found")

        return {
            "skills": parsed.get("skills", []),
            "designation": parsed.get("designation", ""),
            "text": " ".join(parsed.get("skills", []))
        }
    except Exception as e:
        print("❌ pyresparser failed, fallback to manual parsing:", e)
        with pdfplumber.open(path) as pdf:
            text = "\n".join(p.extract_text() for p in pdf.pages if p.extract_text())

        return {
            "skills": [],
            "designation": "",
            "text": text
        }

def search_jobs_from_internet(query):
    try:
        with DDGS() as ddgs:
            results = ddgs.text(
                f"{query} site:linkedin.com/jobs OR site:stepstone.de OR site:indeed.com",
                max_results=10
            )
            return list(results)
    except Exception as e:
        print("❌ DuckDuckGo search failed:", e)
        return [{
            "title": "⚠️ DuckDuckGo Rate Limit erreicht",
            "href": "#",
            "body": "Zu viele Anfragen. Bitte versuchen Sie es später erneut."
        }]


@app.route("/job-matching", methods=["POST"])
def match_jobs():
    if "resume" not in request.files:
        return jsonify({"error": "Resume file missing"}), 400

    file = request.files["resume"]
    path = os.path.join("uploads", file.filename)
    os.makedirs("uploads", exist_ok=True)
    file.save(path)

    resume_data = extract_resume_data(path)
    search_query = f"{resume_data.get('designation', '')} {' '.join(resume_data.get('skills', []))}"

    search_results = search_jobs_from_internet(search_query)


    if search_results and "Rate Limit" in search_results[0]["title"]:
        return jsonify({"matches": search_results})


    kw_model = KeyBERT()
    keywords = kw_model.extract_keywords(resume_data["text"], top_n=10)
    keyword_list = [k[0] for k in keywords]

    matched_jobs = []
    for job in search_results:
        job_text = f"{job.get('title', '')} {job.get('body', '')}"
        vect = CountVectorizer().fit_transform([" ".join(keyword_list), job_text])
        score = cosine_similarity(vect)[0][1] * 100

        if score > 30:
            matched_jobs.append({
                "title": job.get("title", "Unknown"),
                "link": job.get("href", ""),
                "description": job.get("body", ""),
                "match_score": round(score, 2)
            })

    matched_jobs = sorted(matched_jobs, key=lambda x: x["match_score"], reverse=True)
    return jsonify({"matches": matched_jobs})
nlp = spacy.load("en_core_web_sm")

@app.route('/generate_summary', methods=['POST'])
def generate_summary():
    data = request.json
    name = data.get('fullname', 'John Doe')
    job_title = data.get('job_title', 'Software Developer')
    duration = data.get('duration', '5+ years')
    skills = data.get('skills', '')
    keywords = data.get('keywords', [])  # from ATS section
    matched_job = data.get('matched_job', {})  # from job matching section

    keywords_text = ', '.join(keywords)
    match_title = matched_job.get('title', 'your ideal role')

    summary_en = (
        f"{name} is a results-driven {job_title} with {duration} of experience, specializing in {skills}. "
        f"Proven expertise in areas like {keywords_text}. "
        f"Looking to contribute meaningfully in roles like {match_title}, using modern technologies and agile methodologies."
    )

    translator = Translator(to_lang="de")
    summary_de = translator.translate(summary_en)

    return jsonify({
        "summary_en": summary_en,
        "summary_de": summary_de
    })




kw_model = KeyBERT(model="paraphrase-MiniLM-L6-v2")
translator = GoogleTranslator(source="de", target="ar")
text_gen = pipeline("text-generation", model="distilgpt2")
chat_bot = pipeline("text-generation", model="distilgpt2")

@app.route("/lingo")
def lingo_page():
    return render_template("azzlingo.html")

@app.route("/api/lesson")
def generate_lesson():
    level = request.args.get("level", "A1")
    lesson_index = int(request.args.get("lesson", 0))

    prompts_by_level = {
        "A1": "Nenne einfache Sätze für Deutschlernende der Stufe A1:",
        "A2": "Schreibe einen kleinen Text für Niveau A2:",
        "B1": "Gib mir einige typische Sätze für B1-Deutschlernende:",
        "B2": "Erstelle ein Beispiel für einen B2-Text auf Deutsch:"
    }
    prompt = prompts_by_level.get(level, prompts_by_level["A1"])
    generated_text = text_gen(prompt, max_length=100, do_sample=True)[0]['generated_text']

    keywords = kw_model.extract_keywords(generated_text, top_n=4)
    vocab = []
    for keyword, _ in keywords:
        try:
            arabic = translator.translate(keyword)
        except:
            arabic = "(ترجمة غير متوفرة)"
        vocab.append({
            "german": keyword,
            "arabic": arabic,
            "example": f"{keyword} ist wichtig."
        })

    base_word = vocab[0]['german'] if vocab else 'der'
    grammar_text = text_gen(f"Erkläre die deutsche Grammatikregel zu '{base_word}':", max_length=80, do_sample=True)[0]['generated_text']

    grammar_point = {
        "title": "قاعدة ذكية مولدة",
        "description": grammar_text,
        "examples": [f"{word['german']} ist ein Beispiel." for word in vocab]
    }

    return jsonify({
        "title": f"الدرس {lesson_index + 1} لمستوى {level}",
        "vocabulary": vocab,
        "grammar": [grammar_point]
    })








@app.route("/invest_ai")
def invest():
    return render_template("invest.html")
# ===== API: ADD TRANSACTION =====
@app.route("/api/add_transaction", methods=["POST"])
def add_transaction():
    data = request.get_json()
    new_tx = FinanceEntry(
        user_id=g.user.id,  # تأكد أن المستخدم مسجل الدخول
        type=data["type"],
        amount=data["amount"],
        category=data["category"],
        description=data.get("description", "")
    )
    db.session.add(new_tx)
    db.session.commit()
    return jsonify({"status": "success"})


# ===== API: SUMMARY =====
@app.route("/api/summary")
def summary():
    if not g.user:
        return jsonify({"error": "Unauthorized"}), 401

    transactions = FinanceEntry.query.filter_by(user_id=g.user.id).all()
    income = sum(t.amount for t in transactions if t.type == "income")
    expense = sum(t.amount for t in transactions if t.type == "expense")
    investment = sum(t.amount for t in transactions if t.type == "investment")
    balance = income - expense + investment
    return jsonify({
        "balance": balance,
        "income": income,
        "expense": expense,
        "investment": investment
    })


# ===== AI: ANALYZE PORTFOLIO () =====
@app.route("/api/analyze")
def analyze_portfolio():
    if not g.user:
        return jsonify({"error": "Unauthorized"}), 401

    # ✅ جلب المعاملات الاستثمارية للمستخدم الحالي فقط
    transactions = FinanceEntry.query.filter_by(user_id=g.user.id, type="investment").all()

    total = sum(t.amount for t in transactions)
    if total == 0:
        return jsonify({"error": "No investment data"}), 400

    stock = total * 0.6
    bond = total * 0.25
    real_estate = total * 0.10
    crypto = total * 0.05

    growth_rate = round(np.log1p(stock + crypto) * 3 + 5, 2)

    return jsonify({
        "risk": "Moderate",
        "stocks": f"{(stock / total * 100):.1f}%",
        "bonds": f"{(bond / total * 100):.1f}%",
        "real_estate": f"{(real_estate / total * 100):.1f}%",
        "crypto": f"{(crypto / total * 100):.1f}%",
        "growth": f"{growth_rate}%",
        "recommendations": [
            "Increase international ETFs",
            "Adjust crypto to < 5%",
            "Monitor tech exposure quarterly"
        ]
    })

# ===== AI: STOCK PREDICTION USING REAL DATA =====
@app.route("/api/predict", methods=["POST"])
def predict_stock():
    data = request.get_json()
    symbol = data.get("symbol")
    if not symbol:
        return jsonify({"error": "Missing symbol"}), 400

    stock = yf.Ticker(symbol)
    hist = stock.history(period="6mo")

    if hist.empty or len(hist) < 30:
        return jsonify({"error": "Not enough data for this stock"}), 400

    # Prepare data for simple regression
    hist = hist.reset_index()
    hist['day_num'] = np.arange(len(hist))
    X = hist[['day_num']]
    y = hist['Close']

    model = LinearRegression()
    model.fit(X, y)

    future_day = [[len(hist) + 30]]
    pred_price = model.predict(future_day)[0]
    current_price = y.iloc[-1]
    growth = round((pred_price - current_price) / current_price * 100, 2)

    return jsonify({
        "symbol": symbol.upper(),
        "current_price": round(current_price, 2),
        "predicted_price": round(pred_price, 2),
        "growth_estimate": f"{growth}%",
        "recommendation": "BUY" if growth > 3 else "HOLD",
        "confidence": "High" if growth > 5 else "Medium"
    })

if __name__ == "__main__":
    app.run(debug=True)
