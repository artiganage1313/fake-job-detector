import re, joblib, requests
import streamlit as st
from bs4 import BeautifulSoup

st.set_page_config(page_title="Fake Job Detector", page_icon="🔍")

@st.cache_resource
def load():
    return joblib.load("model.pkl"), joblib.load("tfidf.pkl"), joblib.load("threshold.pkl")

model, tfidf, THRESHOLD = load()

def clean_text(t):
    t = t.lower()
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"http\S+", " ", t)
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return re.sub(r"\bamp\b", " ", t)

def fetch_text(url):
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()
    return soup.get_text(" ", strip=True)

st.title("🔍 Fake Job Detector")
st.write("Job cha link kiva text paste kar. Mi sangto te safe aahe ki fake.")

url = st.text_input("Job cha link (optional)")
text = st.text_area("kiva job cha text paste kar", height=250)

if st.button("Check kar"):
    content = text.strip()
    if not content and url.strip():
        try:
            content = fetch_text(url.strip())
        except Exception:
            st.error("Link ughadta ala nahi (kahi sites block kartat). Job cha text paste kar.")
            st.stop()
    if len(content) < 200:
        st.warning("Text khup kami aahe. Purna job description paste kar.")
        st.stop()

    p = model.predict_proba(tfidf.transform([clean_text(content)]))[0][1]
    if p >= THRESHOLD:
        st.error(f"⚠️ FAKE (fake probability: {p*100:.0f}%)")
    else:
        st.success(f"✅ SAFE (fake probability: {p*100:.0f}%)")
    st.caption("He ek ML model cha andaaj aahe, 100% khatri nahi. Apply karnyaadhi company chi swatah chaukashi kar.")
