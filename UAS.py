import os
import time
import re
import requests
import textwrap
import pandas as pd
import torch
import networkx as nx
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from collections import Counter
from selenium import webdriver
from wordcloud import WordCloud
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from transformers import pipeline
from dotenv import load_dotenv
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# ==========================================
# 1. KONFIGURASI & VARIABEL (MISI PAK MENTERI)
# ==========================================
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
USERNAME_IG = os.getenv("IG_USER")
PASSWORD_IG = os.getenv("IG_PASS")

URL_PENCARIAN = "https://search.kompas.com/search?q=dolar+tembus+18000"
URL_IG_POST = "https://www.instagram.com/p/DZWCCeuARD7/"

MAX_COMMENTS_PER_POST = 100

factory = StopWordRemoverFactory()
default_stopwords = set(factory.get_stop_words())
custom_additions = {
    'kompas', 'kompasiana', 'berita', 'terkini', 'com', 'id', 'nya', 'yg', 'di', 'ke', 'dari', 'itu', 'ini', 'dan',
    'yang', 'untuk', 'dengan', 'dalam', 'pada', 'bahwa', 'atau', 'juga', 'akan'
}
all_stopwords = default_stopwords | custom_additions

# ==========================================
# 2. LOAD MODEL NLP (INDOBERT)
# ==========================================
device = 0 if torch.cuda.is_available() else -1
print("\n[INFO] Memuat model IndoBERT...")
id_model = pipeline("text-classification", model="crypter70/IndoBERT-Sentiment-Analysis", token=HF_TOKEN, device=device)
label_map = {"LABEL_0": "negative", "LABEL_1": "positive", "negative": "negative", "positive": "positive"}


# ==========================================
# 3. FUNGSI UTILITAS & SCRAPING ARTIKEL
# ==========================================
def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', ' ', text.lower())
    return text.split()


def filter_text(words):
    return [word for word in words if word not in all_stopwords and len(word) > 2]


def get_top_words(words, top_n=10):
    return Counter(words).most_common(top_n)


def scrape_search_kompas(query_url):
    print(f"\n[INFO] Mencari artikel di: {query_url}")
    try:
        response = requests.get(query_url)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Mengambil link dari hasil pencarian (sesuaikan dengan class CSS search kompas)
        links = []
        for a in soup.find_all('a', href=True):
            if 'kompas.com/read/' in a['href']:
                links.append(a['href'])

        links = list(set(links))  # Hapus duplikat
        print(f"[INFO] Ditemukan {len(links)} artikel. Mengambil isi 10 artikel teratas...")

        all_text = ""
        for link in links[:10]:  # Batasi 10 artikel agar tidak terlalu lama
            try:
                res = requests.get(link)
                s = BeautifulSoup(res.text, 'html.parser')
                content = s.find('div', class_='read__content')
                if content:
                    all_text += " " + ' '.join(p.text for p in content.find_all('p'))
            except:
                continue
        return all_text
    except Exception as e:
        print(f"Error scraping pencarian: {e}")
        return ""


# ==========================================
# 4. SCRAPING KOMENTAR INSTAGRAM (OTOMATIS)
# ==========================================
def scrape_instagram_comments(url_post, username, password):
    options = webdriver.ChromeOptions()
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    driver = webdriver.Chrome(options=options)
    wait = WebDriverWait(driver, 20)

    all_valid_comments = set()

    try:
        print("\n[INFO] TAHAP 1: LOGIN INSTAGRAM (OTOMATIS)")
        driver.get("https://www.instagram.com/accounts/login/")
        time.sleep(5)

        try:
            user_input = driver.find_element(By.NAME, 'email')
            pass_input = driver.find_element(By.NAME, 'pass')
        except:
            user_input = driver.find_element(By.NAME, 'username')
            pass_input = driver.find_element(By.NAME, 'password')

        user_input.send_keys(username)
        pass_input.send_keys(password)
        pass_input.send_keys(Keys.RETURN)

        print("[INFO] Berhasil memasukkan kredensial. Menunggu proses login...")
        time.sleep(7)

        print(f"\n[INFO] TAHAP 2: MEMBUKA POSTINGAN {url_post}")
        driver.get(url_post)
        time.sleep(5)

        scroll_selector = '.x5yr21d.xw2csxc.x1odjw0f.x1n2onr6'
        try:
            scroll_div = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, scroll_selector)))
        except:
            scroll_div = None

        last_height = driver.execute_script("return arguments[0].scrollHeight", scroll_div) if scroll_div else 0
        comments_collected = 0

        while comments_collected < MAX_COMMENTS_PER_POST:
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            spans = soup.find_all('span')

            for span in spans:
                text = span.get_text(strip=True)
                if len(text.split()) > 2 and len(text) > 10:
                    if text not in all_valid_comments:
                        all_valid_comments.add(text)
                        comments_collected += 1

            if comments_collected >= MAX_COMMENTS_PER_POST or not scroll_div:
                break

            driver.execute_script("arguments[0].scrollTop = arguments[0].scrollHeight", scroll_div)
            time.sleep(3)
            new_height = driver.execute_script("return arguments[0].scrollHeight", scroll_div)
            if new_height == last_height:
                break
            last_height = new_height

        print(f"[INFO] Berhasil mengekstrak {len(all_valid_comments)} komentar dari IG.")
        return list(all_valid_comments)

    except Exception as e:
        print(f"Error pada proses Selenium IG: {e}")
        return list(all_valid_comments)
    finally:
        driver.quit()


# ==========================================
# 5. EKSEKUSI PIPELINE & NETWORKX
# ==========================================
# A. Ekstraksi Artikel dari Pencarian Kompas
article_text = scrape_search_kompas(URL_PENCARIAN)

# --- BAGIAN PERBAIKAN: Memotong teks panjang agar tersusun ke bawah secara rapi ---
print("[INFO] Menyusun dan merapikan teks artikel secara vertikal...")
sentences = re.split(r'(?<=\.)\s+', article_text.strip())

with open('paragraf_artikel.txt', 'w', encoding="utf-8") as f:
    for sentence in sentences:
        if len(sentence) > 5:
            # Membungkus kalimat agar otomatis ganti baris jika melewati 100 karakter
            wrapped_text = textwrap.fill(sentence, width=100)
            f.write(wrapped_text + "\n\n")  # \n\n memberikan spasi enter antar paragraf/kalimat
# --- AKHIR DARI PERBAIKAN ---

# Proses filter untuk Keyword & NetworkX
article_words = filter_text(clean_text(article_text))
article_keywords = get_top_words(article_words, 10)

# --- PEMBUATAN GRAF JARINGAN KATA ARTIKEL (NETWORKX) ---
print("\n[INFO] Membangun Jaringan Kata (NetworkX) untuk Artikel...")
G_media = nx.Graph()
# Mengambil 50 kata pertama setelah difilter
words_for_graph = article_words[:50]

for i in range(len(words_for_graph) - 1):
    kata1 = words_for_graph[i]
    kata2 = words_for_graph[i + 1]

    if G_media.has_edge(kata1, kata2):
        G_media[kata1][kata2]['weight'] += 1
    else:
        G_media.add_edge(kata1, kata2, weight=1)

# Sentimen Artikel
article_sentences = [s.strip() for s in article_text.split('.') if len(s.strip()) > 20]
media_sentiments = []

print("[INFO] Menganalisis sentimen artikel...")
for sent in article_sentences:
    try:
        safe_text = sent[:400]
        res = id_model(safe_text)[0]
        mapped_label = label_map.get(res["label"], res["label"]).lower()
        media_sentiments.append({"teks": safe_text, "sumber": "Media (Kompas)", "sentimen": mapped_label})
    except Exception as e:
        pass

# B. Ekstraksi Instagram Data
ig_comments = scrape_instagram_comments(URL_IG_POST, USERNAME_IG, PASSWORD_IG)
public_sentiments = []
all_ig_text = ""

if ig_comments:
    print("\n[INFO] Menganalisis sentimen komentar IG...")
    for c in ig_comments:
        all_ig_text += " " + c
        try:
            safe_text = c[:400]
            res = id_model(safe_text)[0]
            mapped_label = label_map.get(res["label"], res["label"]).lower()
            public_sentiments.append({"teks": c, "sumber": "Publik (Instagram)", "sentimen": mapped_label})
        except Exception as e:
            pass

ig_words = filter_text(clean_text(all_ig_text))
ig_keywords = get_top_words(ig_words, 10)

# ==========================================
# 6. EXPORT CSV & VISUALISASI PLOT
# ==========================================
all_data = media_sentiments + public_sentiments

if len(all_data) == 0:
    print("\n[!] GAGAL MEMBUAT GRAFIK: Data kosong.")
else:
    # 1. Simpan Dataset CSV
    df_all = pd.DataFrame(all_data)
    df_all.to_csv("dataset_sentimen_dolar_rp180001.csv", index=False)
    print("\n[INFO] Dataset tersimpan: 'dataset_sentimen_dolar_rp180001.csv'")
    print("[INFO] Teks artikel tersimpan: 'paragraf_artikel.txt'")

    # 2. Cetak Top Keywords
    print("\n" + "=" * 40)
    print("TOP 10 KATA KUNCI MEDIA (ARTIKEL)")
    print("=" * 40)
    for w, f in article_keywords: print(f"{w}: {f}")

    print("\n" + "=" * 40)
    print("TOP 10 KATA KUNCI PUBLIK (INSTAGRAM)")
    print("=" * 40)
    for w, f in ig_keywords: print(f"{w}: {f}")

    # 3. Visualisasi 1: Jaringan Kata (NetworkX)
    plt.figure(1, figsize=(10, 6))
    pos = nx.spring_layout(G_media, k=0.5)
    weights = [G_media[u][v]['weight'] for u, v in G_media.edges()]

    nx.draw(
        G_media, pos, with_labels=True, node_size=600,
        font_size=9, node_color='lightblue', width=weights, edge_color='gray'
    )
    plt.title("Visualisasi Jaringan Kata Media (Artikel)", fontweight='bold')
    plt.savefig("jaringan_kata_artikel.png")

    # 4. Visualisasi 2: Bar Chart Sentimen Komparatif
    df_media = df_all[df_all['sumber'] == 'Media (Kompas)']
    df_publik = df_all[df_all['sumber'] == 'Publik (Instagram)']

    media_counts = df_media['sentimen'].value_counts()
    publik_counts = df_publik['sentimen'].value_counts()

    plt.figure(2, figsize=(12, 5))
    colors_map = {'negative': 'red', 'positive': 'green'}

    ax1 = plt.subplot(1, 2, 1)
    if not media_counts.empty:
        media_colors = [colors_map.get(x, 'gray') for x in media_counts.index]
        media_counts.plot(kind='bar', ax=ax1, color=media_colors)
        ax1.tick_params(axis='x', rotation=0)
    ax1.set_title("Sentimen Narasi Media")

    ax2 = plt.subplot(1, 2, 2)
    if not publik_counts.empty:
        publik_colors = [colors_map.get(x, 'gray') for x in publik_counts.index]
        publik_counts.plot(kind='bar', ax=ax2, color=publik_colors)
        ax2.tick_params(axis='x', rotation=0)
    ax2.set_title("Sentimen Opini Publik (IG)")

    plt.suptitle("Kesenjangan Narasi: Media vs Publik (Dolar Rp18.000)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("grafik_sentimen_dolar.png")

    # 4. Visualisasi 3: WORD CLOUD

    plt.figure(4, figsize=(14, 6))

    plt.subplot(1, 2, 1)
    wc_media = WordCloud(
        width=800,
        height=400,
        background_color='white'
    ).generate(" ".join(article_words))

    plt.imshow(wc_media, interpolation='bilinear')
    plt.axis('off')
    plt.title("Word Cloud Media")

    plt.subplot(1, 2, 2)
    wc_public = WordCloud(
        width=800,
        height=400,
        background_color='white'
    ).generate(" ".join(ig_words))

    plt.imshow(wc_public, interpolation='bilinear')
    plt.axis('off')
    plt.title("Word Cloud Publik")

    plt.tight_layout()
    plt.savefig("wordcloud_media_vs_publik.png", dpi=300)

    print(
        "\n[INFO] Semua grafik berhasil digambar. Menampilkan jendela grafik (Tutup jendela untuk mengakhiri program)... 🎉")
    plt.show()