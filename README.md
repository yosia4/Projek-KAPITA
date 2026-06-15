# Analisis Sentimen Media vs Opini Publik Instagram

## Deskripsi

Proyek ini bertujuan membandingkan sentimen narasi media mengenai isu **Dolar Rp18.000** dengan opini publik pada komentar Instagram menggunakan model **IndoBERT Sentiment Analysis**.

##Clone repository
```bash
pip install -r requirements.txt
```
## Instalasi

Install seluruh library:

```bash
pip install -r requirements.txt
```

## Konfigurasi

Buat file `.env`:

```env
HF_TOKEN=token_huggingface_anda
IG_USER=username_instagram
IG_PASS=password_instagram
```

## Menjalankan Program

```bash
python main.py
```

## Output

Program akan menghasilkan:

* `dataset_sentimen_dolar_rp180001.csv`
* `paragraf_artikel.txt`
* `jaringan_kata_artikel.png`
* `grafik_sentimen_dolar.png`
* `wordcloud_media_vs_publik.png`

## Teknologi

* Python
* Selenium
* BeautifulSoup
* IndoBERT
* NetworkX
* Matplotlib
* Pandas
