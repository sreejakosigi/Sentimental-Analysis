import streamlit as st
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from youtube_comment_downloader import YoutubeCommentDownloader, SORT_BY_POPULAR
from textblob import TextBlob
import matplotlib.pyplot as plt
import re
import hashlib
import sqlite3
import pandas as pd
from datetime import datetime

# Page config
st.set_page_config(page_title="YouTube Sentiment Analyzer", page_icon="📊", layout="wide")

# ────────────────────────────────────────────────
# Database Setup
# ────────────────────────────────────────────────
def init_db():
    """Initialize SQLite database for user management"""
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    
    # Create users table
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE NOT NULL,
                  password TEXT NOT NULL,
                  email TEXT,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    
    # Create analysis history table
    c.execute('''CREATE TABLE IF NOT EXISTS analysis_history
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER,
                  video_url TEXT,
                  video_id TEXT,
                  analysis_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  transcript_sentiment_score REAL,
                  comments_sentiment_score REAL,
                  FOREIGN KEY (user_id) REFERENCES users (id))''')
    
    conn.commit()
    conn.close()

def hash_password(password):
    """Hash password using SHA256"""
    return hashlib.sha256(password.encode()).hexdigest()

def create_user(username, password, email=None):
    """Create a new user"""
    try:
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute("INSERT INTO users (username, password, email) VALUES (?, ?, ?)",
                  (username, hash_password(password), email))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False

def authenticate_user(username, password):
    """Authenticate user"""
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT id, username FROM users WHERE username = ? AND password = ?",
              (username, hash_password(password)))
    user = c.fetchone()
    conn.close()
    return user

def save_analysis_history(user_id, video_url, video_id, transcript_score, comments_score):
    """Save analysis to history"""
    if user_id:
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute("""INSERT INTO analysis_history 
                     (user_id, video_url, video_id, transcript_sentiment_score, comments_sentiment_score)
                     VALUES (?, ?, ?, ?, ?)""",
                  (user_id, video_url, video_id, transcript_score, comments_score))
        conn.commit()
        conn.close()

def get_user_history(user_id):
    """Get analysis history for a user"""
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("""SELECT video_url, video_id, analysis_date, 
                        transcript_sentiment_score, comments_sentiment_score
                 FROM analysis_history 
                 WHERE user_id = ? 
                 ORDER BY analysis_date DESC 
                 LIMIT 20""", (user_id,))
    history = c.fetchall()
    conn.close()
    return history

# ────────────────────────────────────────────────
# Authentication UI
# ────────────────────────────────────────────────
def login_page():
    """Display login page"""
    st.title("🔐 YouTube Sentiment Analyzer")
    
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        st.subheader("Login to your account")
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        
        if st.button("Login", type="primary"):
            if username and password:
                user = authenticate_user(username, password)
                if user:
                    st.session_state['user_id'] = user[0]
                    st.session_state['username'] = user[1]
                    st.session_state['logged_in'] = True
                    st.success(f"Welcome back, {username}!")
                    st.rerun()
                else:
                    st.error("Invalid username or password")
            else:
                st.warning("Please enter both username and password")
    
    with tab2:
        st.subheader("Create a new account")
        new_username = st.text_input("Username", key="reg_username")
        new_email = st.text_input("Email (optional)", key="reg_email")
        new_password = st.text_input("Password", type="password", key="reg_password")
        confirm_password = st.text_input("Confirm Password", type="password", key="reg_confirm")
        
        if st.button("Register"):
            if new_username and new_password:
                if new_password != confirm_password:
                    st.error("Passwords do not match")
                elif len(new_password) < 6:
                    st.error("Password must be at least 6 characters")
                else:
                    if create_user(new_username, new_password, new_email):
                        st.success("Account created successfully! Please login.")
                    else:
                        st.error("Username already exists")
            else:
                st.warning("Please fill in all required fields")

def logout():
    """Logout user"""
    for key in ['user_id', 'username', 'logged_in']:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()

# ────────────────────────────────────────────────
# Helper Functions
# ────────────────────────────────────────────────
def extract_video_id(url: str):
    if not url:
        return None
    parsed = urlparse(url.strip())
    if parsed.hostname in ['youtu.be', 'www.youtu.be']:
        return parsed.path.lstrip('/')
    if 'youtube.com' in parsed.hostname:
        query = parse_qs(parsed.query)
        return query.get('v', [None])[0]
    return None

def get_transcript(video_id: str):
    """
    Get transcript using the updated youtube-transcript-api (v1.0.0+)
    """
    try:
        ytt_api = YouTubeTranscriptApi()
        transcript_list = ytt_api.list(video_id)
        
        try:
            transcript = transcript_list.find_transcript(['en', 'en-US', 'en-GB'])
        except NoTranscriptFound:
            available_transcripts = list(transcript_list)
            if not available_transcripts:
                return "", 0
            transcript = available_transcripts[0]
            st.info(f"Using transcript in language: {transcript.language}")
        
        fetched_transcript = transcript.fetch()
        full_text = " ".join(snippet.text for snippet in fetched_transcript)
        return full_text, len(fetched_transcript)
    
    except TranscriptsDisabled:
        st.warning("⚠️ Transcripts are disabled for this video.")
        return "", 0
    except NoTranscriptFound:
        st.warning("⚠️ No transcript found for this video.")
        return "", 0
    except Exception as e:
        st.error(f"Transcript error: {str(e)}")
        return "", 0

def analyze_sentiment(text_list, source_name=""):
    """Returns 7 values always for safe unpacking"""
    if not text_list:
        return 0, 0, 0, 0, 0.0, 0.0, 0.0

    positive = negative = neutral = 0

    for text in text_list:
        try:
            polarity = TextBlob(text).sentiment.polarity
            if polarity > 0.05:
                positive += 1
            elif polarity < -0.05:
                negative += 1
            else:
                neutral += 1
        except:
            continue

    total = positive + negative + neutral
    if total == 0:
        return 0, 0, 0, 0, 0.0, 0.0, 0.0

    pos_pct = (positive / total) * 100
    neg_pct = (negative / total) * 100
    neu_pct = (neutral / total) * 100

    return positive, negative, neutral, total, pos_pct, neg_pct, neu_pct

# ────────────────────────────────────────────────
# Main Application
# ────────────────────────────────────────────────
def main_app():
    # Sidebar with user info
    with st.sidebar:
        st.header(f"👋 Welcome, {st.session_state['username']}!")
        
        if st.button("🚪 Logout", type="secondary"):
            logout()
        
        st.divider()
        st.header("⚙️ Settings")
        
        url = st.text_input(
            "YouTube Video URL",
            placeholder="https://www.youtube.com/watch?v=WHYMGNbPv2U"
        )
        
        max_comments = st.slider("Max comments to analyze", 50, 1000, 300, 50)
        sort_by_option = st.radio("Sort comments by", ["Popular", "Recent"])
        analyze_button = st.button("🚀 Analyze Video", type="primary")
        
        # Show history in sidebar
        st.divider()
        st.header("📜 Recent Analyses")
        history = get_user_history(st.session_state['user_id'])
        if history:
            for item in history[:5]:
                with st.expander(f"📹 {item[1][:30]}..."):
                    st.write(f"Date: {item[2][:16]}")
                    st.write(f"Transcript Score: {item[3]:.1f}")
                    st.write(f"Comments Score: {item[4]:.1f}")
        else:
            st.info("No analysis history yet")

    # Main content
    st.title("📊 YouTube Video Sentiment Analyzer")
    st.markdown("Analyze **transcript** and **top comments** sentiment from any YouTube video.")

    if analyze_button and url:
        video_id = extract_video_id(url)
        if not video_id:
            st.error("❌ Invalid YouTube URL. Could not extract Video ID.")
            st.stop()

        st.success(f"✅ Video ID extracted: **{video_id}**")

        progress_bar = st.progress(0)
        status = st.empty()

        # 1. Fetch Transcript
        status.text("Fetching transcript...")
        progress_bar.progress(25)
        
        full_text, num_sentences = get_transcript(video_id)

        if full_text:
            st.info(f"✅ Transcript fetched successfully (~{num_sentences} segments, {len(full_text):,} characters)")
        else:
            st.warning("⚠️ No transcript available for sentiment analysis. Only comments will be analyzed.")

        progress_bar.progress(50)

        # 2. Fetch Comments
        status.text("Fetching comments...")
        progress_bar.progress(65)

        downloader = YoutubeCommentDownloader()
        comments = []

        try:
            sort_const = SORT_BY_POPULAR if sort_by_option == "Popular" else None
            generator = downloader.get_comments_from_url(url, sort_by=sort_const)
            
            for comment in generator:
                if comment and comment.get('text', '').strip():
                    comments.append(comment['text'])
                if len(comments) >= max_comments:
                    break

            st.success(f"✅ Fetched **{len(comments)}** comments")
        except Exception as e:
            st.warning(f"⚠️ Could not fetch comments: {str(e)}")
            comments = []

        progress_bar.progress(85)

        # 3. Sentiment Analysis
        status.text("Analyzing sentiment...")

        # Split transcript into sentences
        transcript_sentences = []
        if full_text:
            sentences = re.split(r'[.!?]+', full_text)
            transcript_sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        # Analyze transcript sentiment
        pos_t, neg_t, neu_t, total_t, pos_pct_t, neg_pct_t, neu_pct_t = analyze_sentiment(transcript_sentences)

        # Analyze comments sentiment
        pos_c, neg_c, neu_c, total_c, pos_pct_c, neg_pct_c, neu_pct_c = analyze_sentiment(comments)

        # Calculate overall scores for history
        transcript_score = pos_pct_t - neg_pct_t if total_t > 0 else 0
        comments_score = pos_pct_c - neg_pct_c if total_c > 0 else 0
        
        # Save to history
        save_analysis_history(st.session_state['user_id'], url, video_id, transcript_score, comments_score)

        progress_bar.progress(100)
        status.empty()

        # ────────────────────────────────────────────────
        # Results Display
        # ────────────────────────────────────────────────
        st.header("📈 Sentiment Analysis Results")
        
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📝 Transcript Sentiment")
            if total_t > 0:
                col1_1, col1_2, col1_3 = st.columns(3)
                with col1_1:
                    st.metric("😊 Positive", f"{pos_pct_t:.1f}%", f"{pos_t} sentences")
                with col1_2:
                    st.metric("😠 Negative", f"{neg_pct_t:.1f}%", f"{neg_t} sentences")
                with col1_3:
                    st.metric("😐 Neutral", f"{neu_pct_t:.1f}%", f"{neu_t} sentences")
            else:
                st.info("📭 No transcript data available")

        with col2:
            st.subheader("💬 Comments Sentiment")
            if total_c > 0:
                col2_1, col2_2, col2_3 = st.columns(3)
                with col2_1:
                    st.metric("😊 Positive", f"{pos_pct_c:.1f}%", f"{pos_c} comments")
                with col2_2:
                    st.metric("😠 Negative", f"{neg_pct_c:.1f}%", f"{neg_c} comments")
                with col2_3:
                    st.metric("😐 Neutral", f"{neu_pct_c:.1f}%", f"{neu_c} comments")
            else:
                st.info("📭 No comments data available")

        # Pie Charts
        if total_t > 0 or total_c > 0:
            st.subheader("📊 Sentiment Distribution")
            
            fig, axs = plt.subplots(1, 2, figsize=(12, 5.5))

            labels = ['Positive', 'Negative', 'Neutral']
            colors = ['#4CAF50', '#F44336', '#FFC107']
            
            if total_t > 0:
                sizes_t = [pos_t, neg_t, neu_t]
                axs[0].pie(sizes_t, labels=labels, autopct='%1.1f%%', 
                          colors=colors, startangle=90, explode=(0.05, 0.05, 0.05))
                axs[0].set_title(f"📝 Transcript Sentiment\n({total_t} sentences)", fontsize=12, fontweight='bold')
            else:
                axs[0].text(0.5, 0.5, "No Transcript Data", ha='center', va='center', fontsize=14)
                axs[0].set_title("📝 Transcript Sentiment", fontsize=12, fontweight='bold')
            
            if total_c > 0:
                sizes_c = [pos_c, neg_c, neu_c]
                axs[1].pie(sizes_c, labels=labels, autopct='%1.1f%%',
                          colors=colors, startangle=90, explode=(0.05, 0.05, 0.05))
                axs[1].set_title(f"💬 Comments Sentiment\n({total_c} comments)", fontsize=12, fontweight='bold')
            else:
                axs[1].text(0.5, 0.5, "No Comments Data", ha='center', va='center', fontsize=14)
                axs[1].set_title("💬 Comments Sentiment", fontsize=12, fontweight='bold')

            plt.tight_layout()
            st.pyplot(fig)
        
        # Overall sentiment score
        if total_t > 0 or total_c > 0:
            st.subheader("🎯 Overall Sentiment Score")
            
            overall_score = 0
            if total_t > 0 and total_c > 0:
                overall_score = ((pos_pct_t + pos_pct_c) / 2) - ((neg_pct_t + neg_pct_c) / 2)
            elif total_t > 0:
                overall_score = pos_pct_t - neg_pct_t
            elif total_c > 0:
                overall_score = pos_pct_c - neg_pct_c
                
            sentiment_color = "green" if overall_score > 20 else "red" if overall_score < -20 else "orange"
            
            st.markdown(f"""
            <div style='text-align: center; padding: 20px; background-color: #f0f2f6; border-radius: 10px;'>
                <h3>Sentiment Balance Score</h3>
                <p style='font-size: 48px; color: {sentiment_color}; font-weight: bold; margin: 0;'>
                    {overall_score:.1f}
                </p>
                <p style='color: #666;'>Positive score indicates favorable sentiment, negative score indicates unfavorable sentiment</p>
            </div>
            """, unsafe_allow_html=True)

        # Expanders for raw data
        with st.expander("📜 Full Transcript"):
            if full_text:
                st.text_area("Transcript Text", full_text, height=250, disabled=True)
            else:
                st.write("No transcript available for this video.")

        with st.expander(f"💬 Sample Comments ({len(comments)} fetched)"):
            if comments:
                for i, comment in enumerate(comments[:20], 1):
                    with st.container():
                        st.write(f"**{i}.** {comment}")
                        st.divider()
            else:
                st.write("No comments fetched for this video.")
        
        # Add information about the analysis
        with st.expander("ℹ️ About this analysis"):
            st.markdown("""
            **How sentiment analysis works:**
            - Uses TextBlob library for sentiment analysis
            - Polarity score ranges from -1 (very negative) to +1 (very positive)
            - Classification: Positive (>0.05), Neutral (-0.05 to 0.05), Negative (<-0.05)
            
            **YouTube Transcript API Changes:**
            - Version 1.0.0+ uses a new API: `YouTubeTranscriptApi().fetch()` instead of `get_transcript()`
            - The code has been updated to use the new API pattern
            
            **Limitations:**
            - Sentiment analysis may not capture sarcasm or context
            - Transcript availability depends on video settings
            - YouTube API limitations may affect comment fetching
            """)

    elif analyze_button:
        st.warning("⚠️ Please enter a valid YouTube URL")
    
    st.caption("Built with Streamlit • YouTube Sentiment Analyzer v2.0 • Uses youtube-transcript-api v1.0.0+")

# ────────────────────────────────────────────────
# App Entry Point
# ────────────────────────────────────────────────
def run_app():
    # Initialize database
    init_db()
    
    # Initialize session state
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
    
    # Show login page or main app
    if not st.session_state.get('logged_in', False):
        login_page()
    else:
        main_app()

if __name__ == "__main__":
    run_app()