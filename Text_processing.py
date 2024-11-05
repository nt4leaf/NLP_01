pip install --upgrade google-api-python-client
import re
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
import warnings
warnings.filterwarnings('ignore')
from collections import Counter
from sklearn.metrics import confusion_matrix, classification_report
from keras.preprocessing import sequence

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')

# Hàm để trích xuất ID video từ liên kết YouTube
def extract_video_id(url):
    # Sử dụng regex để tìm ID video
    video_id = re.search(r'(?<=v=)[^&#]+', url)
    if not video_id:
        video_id = re.search(r'(?<=be/)[^&#]+', url)
    return video_id.group(0) if video_id else None

# Giao diện Streamlit
st.title('YouTube Video ID Extractor')

# Nhập liên kết YouTube
url = st.text_input('Enter YouTube URL')

# Hiển thị ID video nếu liên kết hợp lệ
if url:
    video_id = extract_video_id(url)
    if video_id:
        st.success(f'Video ID: {video_id}')
    else:
        st.error('Invalid YouTube URL')
