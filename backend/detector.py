# import os
# import re
# import requests
# import numpy as np
# import pandas as pd
# from PIL import Image
# from urllib.parse import urlparse, parse_qs
# from transformers import pipeline, CLIPProcessor, CLIPModel
# from sklearn.metrics.pairwise import cosine_similarity
# from googleapiclient.discovery import build
# import torch
# import joblib
# from dotenv import load_dotenv
# load_dotenv()
# # ==============================
# # Feature Extractor Class
# # ==============================

# class YouTubeFeatureExtractor:
#     def __init__(self, api_key):
#         self.api_key = api_key
#         self.youtube = build('youtube', 'v3', developerKey=api_key)

#         self.thumbnail_dir = "thumbnails"
#         self.summary_path = "summary.txt"   # put your summary file here if needed

#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         self.clip_model = CLIPModel.from_pretrained(
#             "openai/clip-vit-large-patch14"
#         ).to(self.device)
#         self.processor = CLIPProcessor.from_pretrained(
#             "openai/clip-vit-large-patch14"
#         )

#         os.makedirs(self.thumbnail_dir, exist_ok=True)

#     def extract_video_id(self, url):
#         query = urlparse(url)
#         if query.hostname == 'youtu.be':
#             return query.path[1:]
#         if query.hostname in ('www.youtube.com', 'youtube.com'):
#             if query.path == '/watch':
#                 return parse_qs(query.query)['v'][0]
#             elif query.path.startswith('/embed/'):
#                 return query.path.split('/')[2]
#             elif query.path.startswith('/v/'):
#                 return query.path.split('/')[2]
#         return None

#     def get_video_metadata(self, video_id):
#         request = self.youtube.videos().list(
#             part="snippet,statistics,contentDetails",
#             id=video_id
#         )
#         response = request.execute()
#         return response['items'][0] if response['items'] else None

#     def download_thumbnail(self, url, video_id):
#         image_path = os.path.join(self.thumbnail_dir, f"thumbnail_{video_id}.jpg")
#         response = requests.get(url)
#         with open(image_path, 'wb') as f:
#             f.write(response.content)
#         return image_path

#     def get_video_duration_seconds(self, iso_duration):
#         pattern = re.compile(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?')
#         matches = pattern.match(iso_duration)
#         if not matches:
#             return 0
#         hours, minutes, seconds = matches.groups()
#         hours = int(hours) if hours else 0
#         minutes = int(minutes) if minutes else 0
#         seconds = int(seconds) if seconds else 0
#         return hours * 3600 + minutes * 60 + seconds

#     def load_summary(self):
#         try:
#             with open(self.summary_path, 'r') as f:
#                 return f.read().strip()
#         except:
#             return ""

#     def truncate_text(self, text, max_tokens=77):
#         return ' '.join(text.split()[:max_tokens])

#     def get_text_embedding(self, text):
#         inputs = self.processor(
#             text=self.truncate_text(text),
#             return_tensors="pt",
#             padding=True,
#             truncation=True,
#             max_length=77
#         ).to(self.device)

#         with torch.no_grad():
#             return self.clip_model.get_text_features(**inputs).cpu().numpy()

#     def get_image_embedding(self, image_path):
#         image = Image.open(image_path).convert("RGB")
#         inputs = self.processor(images=image, return_tensors="pt").to(self.device)

#         with torch.no_grad():
#             return self.clip_model.get_image_features(**inputs).cpu().numpy()

#     def cosine_similarity_score(self, emb1, emb2):
#         return cosine_similarity(emb1, emb2)[0][0]

#     def get_comments(self, video_id, max_comments=100):
#         comments = []
#         request = self.youtube.commentThreads().list(
#             part="snippet",
#             videoId=video_id,
#             maxResults=100,
#             textFormat="plainText"
#         )
#         response = request.execute()

#         while request and len(comments) < max_comments:
#             for item in response['items']:
#                 comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
#                 comments.append(comment)
#                 if len(comments) >= max_comments:
#                     break
#             if 'nextPageToken' in response and len(comments) < max_comments:
#                 request = self.youtube.commentThreads().list_next(request, response)
#                 response = request.execute()
#             else:
#                 break
#         return comments

#     def analyze_sentiments(self, comments):
#         sentiment_pipeline = pipeline("sentiment-analysis")
#         results = sentiment_pipeline(comments, truncation=True)

#         sentiment_scores = []
#         positive_count, negative_count = 0, 0

#         for result in results:
#             score = result['score']
#             label = result['label']
#             sentiment_scores.append(score if label == 'POSITIVE' else -score)
#             if label == 'POSITIVE':
#                 positive_count += 1
#             else:
#                 negative_count += 1

#         avg_sentiment = np.mean(sentiment_scores)
#         total = len(comments)

#         return {
#             "Avg_Sentiment_Score": round(avg_sentiment, 3),
#             "Positive_Comment_Ratio": round(positive_count / total, 3),
#             "Negative_Comment_Ratio": round(negative_count / total, 3),
#             "Total_Comments_Analyzed": total
#         }

#     def analyze_video(self, video_url):
#         video_id = self.extract_video_id(video_url)
#         metadata = self.get_video_metadata(video_id)

#         if not metadata:
#             return None

#         snippet = metadata['snippet']
#         stats = metadata['statistics']
#         content_details = metadata['contentDetails']

#         title = snippet['title']
#         views = int(stats.get('viewCount', 0))
#         likes = int(stats.get('likeCount', 0))
#         dislikes = int(stats.get('dislikeCount', 1))

#         duration = self.get_video_duration_seconds(content_details['duration'])
#         avg_watch_time = duration * 0.65
#         engagement_ratio = (likes + dislikes) / views if views else 0
#         view_like_ratio = views / likes if likes else 0
#         view_dislike_ratio = views / dislikes if dislikes else 0

#         thumbnail_url = snippet['thumbnails']['high']['url']
#         image_path = self.download_thumbnail(thumbnail_url, video_id)

#         summary = self.load_summary()
#         title_emb = self.get_text_embedding(title)
#         summary_emb = self.get_text_embedding(summary)
#         image_emb = self.get_image_embedding(image_path)

#         title_thumb_sim = round(self.cosine_similarity_score(title_emb, image_emb), 3)
#         summary_thumb_sim = round(self.cosine_similarity_score(summary_emb, image_emb), 3)
#         title_summary_sim = round(self.cosine_similarity_score(title_emb, summary_emb), 3)

#         comments = self.get_comments(video_id, max_comments=100)
#         sentiment = self.analyze_sentiments(comments)

#         features = {
#             "Views": views,
#             "Likes": likes,
#             "Dislikes": dislikes,
#             "View-to-Like Ratio": round(view_like_ratio, 3),
#             "View-to-Dislike Ratio": round(view_dislike_ratio, 3),
#             "Avg Watch Time": round(avg_watch_time, 2),
#             "Video Duration (s)": duration,
#             "Engagement Ratio": round(engagement_ratio, 3),
#             "Title-Thumbnail Similarity": title_thumb_sim,
#             "Summary-Thumbnail Similarity": summary_thumb_sim,
#             "Title-Summary Similarity": title_summary_sim,
#             **sentiment
#         }

#         return pd.DataFrame([features])


# # ==============================
# # Prediction Function
# # ==============================

# # API_KEY = "AIzaSyDoHflAnplNt8nCUU25ryveOF8hW-zSF3s"
# API_KEY = os.getenv("YOUTUBE_API_KEY")

# MODEL_DIR = "models"

# model_files = [
#     'logistic_regression_model.pkl',
#     'xgboost_model.pkl',
#     'gradient_boosting_model.pkl',
#     'random_forest_model.pkl',
#     'decision_tree_model.pkl'
# ]

# models = [joblib.load(os.path.join(MODEL_DIR, f)) for f in model_files]


# def predict_clickbait(url: str):
#     extractor = YouTubeFeatureExtractor(API_KEY)

#     video_features_df = extractor.analyze_video(url)

#     if video_features_df is None:
#         return "Error"

#     # Drop similarity columns like your code
#     columns_to_remove = [
#         "Title-Thumbnail Similarity",
#         "Summary-Thumbnail Similarity",
#         "Title-Summary Similarity"
#     ]
#     video_features_df = video_features_df.drop(columns=columns_to_remove, errors='ignore')

#     predictions_all = []

#     for model in models:
#         pred = model.predict(video_features_df)
#         predictions_all.append(pred)

#     predictions_array = np.array(predictions_all)

#     final_predictions = []
#     for i in range(predictions_array.shape[1]):
#         values, counts = np.unique(predictions_array[:, i], return_counts=True)
#         final_predictions.append(values[np.argmax(counts)])

#     final = final_predictions[0]

#     return "Clickbait" if final == 0 else "Not Clickbait"

import os
import re
import requests
import numpy as np
import pandas as pd
from PIL import Image
from urllib.parse import urlparse, parse_qs
from transformers import pipeline, CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity
from googleapiclient.discovery import build
import torch
import joblib
from dotenv import load_dotenv

load_dotenv()

# ==============================
# Global setup
# ==============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

THUMB_DIR = os.path.join(BASE_DIR, "thumbnails")
MODEL_DIR = os.path.join(BASE_DIR, "models")
SUMMARY_PATH = os.path.join(BASE_DIR, "summary.txt")

os.makedirs(THUMB_DIR, exist_ok=True)

API_KEY = os.getenv("YOUTUBE_API_KEY")

device = "cuda" if torch.cuda.is_available() else "cpu"

print("🔄 Loading CLIP model...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

print("🔄 Loading sentiment model...")
sentiment_pipeline = pipeline("sentiment-analysis")

print("🔄 Loading ML models...")
model_files = [
    "logistic_regression_model.pkl",
    "xgboost_model.pkl",
    "gradient_boosting_model.pkl",
    "random_forest_model.pkl",
    "decision_tree_model.pkl"
]
models = [joblib.load(os.path.join(MODEL_DIR, f)) for f in model_files]

print("✅ All models loaded!")


# ==============================
# Feature Extractor Class
# ==============================

class YouTubeFeatureExtractor:
    def __init__(self, api_key):
        self.api_key = api_key
        self.youtube = build("youtube", "v3", developerKey=api_key)

    def extract_video_id(self, url):
        query = urlparse(url)
        if query.hostname == "youtu.be":
            return query.path[1:]
        if query.hostname in ("www.youtube.com", "youtube.com"):
            if query.path == "/watch":
                return parse_qs(query.query)["v"][0]
            elif query.path.startswith("/embed/"):
                return query.path.split("/")[2]
            elif query.path.startswith("/v/"):
                return query.path.split("/")[2]
        return None

    def get_video_metadata(self, video_id):
        request = self.youtube.videos().list(
            part="snippet,statistics,contentDetails",
            id=video_id
        )
        response = request.execute()
        return response["items"][0] if response["items"] else None

    def download_thumbnail(self, url, video_id):
        image_path = os.path.join(THUMB_DIR, f"thumbnail_{video_id}.jpg")
        r = requests.get(url, timeout=10)
        with open(image_path, "wb") as f:
            f.write(r.content)
        return image_path

    def get_video_duration_seconds(self, iso_duration):
        pattern = re.compile(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?")
        matches = pattern.match(iso_duration)
        if not matches:
            return 0
        h, m, s = matches.groups()
        return (int(h) if h else 0) * 3600 + (int(m) if m else 0) * 60 + (int(s) if s else 0)

    def truncate_text(self, text, max_tokens=77):
        return " ".join(text.split()[:max_tokens])

    def get_text_embedding(self, text):
        inputs = processor(
            text=self.truncate_text(text),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77
        ).to(device)
        with torch.no_grad():
            return clip_model.get_text_features(**inputs).cpu().numpy()

    def get_image_embedding(self, image_path):
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            return clip_model.get_image_features(**inputs).cpu().numpy()

    def cosine_similarity_score(self, e1, e2):
        return cosine_similarity(e1, e2)[0][0]

    def get_comments(self, video_id, max_comments=100):
        comments = []
        request = self.youtube.commentThreads().list(
            part="snippet", videoId=video_id, maxResults=100, textFormat="plainText"
        )
        response = request.execute()

        while request and len(comments) < max_comments:
            for item in response["items"]:
                c = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                comments.append(c)
                if len(comments) >= max_comments:
                    break
            if "nextPageToken" in response:
                request = self.youtube.commentThreads().list_next(request, response)
                response = request.execute()
            else:
                break
        return comments

    def analyze_sentiments(self, comments):
        results = sentiment_pipeline(comments, truncation=True)

        scores, pos, neg = [], 0, 0
        for r in results:
            s = r["score"]
            if r["label"] == "POSITIVE":
                scores.append(s)
                pos += 1
            else:
                scores.append(-s)
                neg += 1

        total = len(comments)
        return {
            "Avg_Sentiment_Score": round(np.mean(scores), 3),
            "Positive_Comment_Ratio": round(pos / total, 3),
            "Negative_Comment_Ratio": round(neg / total, 3),
            "Total_Comments_Analyzed": total
        }

    def analyze_video(self, url):
        video_id = self.extract_video_id(url)
        meta = self.get_video_metadata(video_id)
        if not meta:
            return None

        snip = meta["snippet"]
        stats = meta["statistics"]
        content = meta["contentDetails"]

        title = snip["title"]
        views = int(stats.get("viewCount", 0))
        likes = int(stats.get("likeCount", 0))
        dislikes = int(stats.get("dislikeCount", 1))

        dur = self.get_video_duration_seconds(content["duration"])
        avg_watch = dur * 0.65
        eng = (likes + dislikes) / views if views else 0
        v_l = views / likes if likes else 0
        v_d = views / dislikes if dislikes else 0

        thumb_url = snip["thumbnails"]["high"]["url"]
        img_path = self.download_thumbnail(thumb_url, video_id)

        title_emb = self.get_text_embedding(title)
        img_emb = self.get_image_embedding(img_path)

        title_thumb_sim = round(self.cosine_similarity_score(title_emb, img_emb), 3)

        comments = self.get_comments(video_id, 100)
        sentiment = self.analyze_sentiments(comments)

        features = {
            "Views": views,
            "Likes": likes,
            "Dislikes": dislikes,
            "View-to-Like Ratio": round(v_l, 3),
            "View-to-Dislike Ratio": round(v_d, 3),
            "Avg Watch Time": round(avg_watch, 2),
            "Video Duration (s)": dur,
            "Engagement Ratio": round(eng, 3),
            "Title-Thumbnail Similarity": title_thumb_sim,
            **sentiment
        }

        return pd.DataFrame([features])


# ==============================
# Prediction Function
# ==============================

def predict_clickbait(url: str):
    print("➡️ Predicting for:", url)

    extractor = YouTubeFeatureExtractor(API_KEY)
    df = extractor.analyze_video(url)

    if df is None:
        return "Error: Could not fetch video data"

    df = df.drop(columns=["Title-Thumbnail Similarity"], errors="ignore")

    preds = []
    for model in models:
        preds.append(model.predict(df))

    preds = np.array(preds)

    values, counts = np.unique(preds[:, 0], return_counts=True)
    final = values[np.argmax(counts)]

    return "Clickbait" if final == 0 else "Not Clickbait"
