import streamlit as st
from googleapiclient.discovery import build
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound

# ------------------------
# 🔐 Secret YouTube API Key
# ------------------------
API_KEY = st.secrets["YOUTUBE_API_KEY"]

# ------------------------
# 🔍 Fetch Videos from YouTube
# ------------------------

@st.cache_data
def get_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([t["text"] for t in transcript])
    except (TranscriptsDisabled, NoTranscriptFound):
        return ""
        
def fetch_videos(query, use_transcripts):
    max_results=10
    youtube = build("youtube", "v3", developerKey=API_KEY)
    request = youtube.search().list(
        part="snippet",
        q=query,
        type="video",
        maxResults=max_results
    )
    response = request.execute()
    videos = []
    for item in response["items"]:
        video_id = item['id']['videoId']
        videos.append({
            "video_id": item["id"]["videoId"],
            "title": item["snippet"]["title"],
            "description": item["snippet"]["description"],
            "transcript": get_transcript(video_id) if use_transcripts else ""
        })
    return videos

# ------------------------
# 🧠 Load SBERT Model
# ------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

# ------------------------
# 🔢 Embed Texts
# ------------------------
def embed_texts(texts, model):
    return model.encode(texts, show_progress_bar=False)

# ------------------------
# 🎯 Run Semantic Search
# ------------------------
def run_search(user_input, model, use_transcripts):
    videos = fetch_videos(user_input, use_transcripts)
    video_texts = [f"{v['title']} {v['description']} {v.get('transcript', '')}" for v in videos]
    video_embeddings = embed_texts(video_texts, model)
    query_embedding = embed_texts([user_input], model)

    nn = NearestNeighbors(n_neighbors=5, metric="cosine")
    nn.fit(video_embeddings)
    distances, indices = nn.kneighbors(query_embedding)

    return [videos[i] for i in indices[0]]

# ------------------------
# 🌐 Streamlit UI
# ------------------------
st.title("YouTube Semantic Search")
st.write("Search for educational YouTube videos using natural language.")

user_input = st.text_input("🔎 What do you want to learn about?")
use_transcripts = st.checkbox("Search video transcripts", value=True)

if user_input:
    with st.spinner("Searching for the most relevant videos..."):
        model = load_model()
        results = run_search(user_input, model, use_transcripts=use_transcripts)

    st.subheader("Top Matching Videos")
    for r in results:
        video_id = r["video_id"]
        url = f"https://www.youtube.com/watch?v={video_id}"
        st.markdown(f"**{r['title']}**")
        st.video(url)
