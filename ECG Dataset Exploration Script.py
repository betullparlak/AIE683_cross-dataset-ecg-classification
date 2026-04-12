"""
ECG Dataset Exploration Script (v2)
====================================
Kullanım: python 01_dataset_exploration_v2.py
"""

# --- 0. KURULUM ---
import subprocess, sys
for pkg in ['wfdb']:
    try:
        __import__(pkg)
    except ImportError:
        print(f"{pkg} kuruluyor...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg])

# --- 1. IMPORTLAR ---
import os
import glob
import numpy as np
import pandas as pd
from collections import Counter
import ast
import warnings
warnings.filterwarnings('ignore')

# --- 2. PATH AYARLARI ---
PATHS = {
    'PTB-XL': r"C:\Users\BETÜL\Desktop\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1",
    'Chapman': r"C:\Users\BETÜL\Desktop\WFDB_ChapmanShaoxing",
    'CPSC2018': r"C:\Users\BETÜL\Desktop\Training_WFDB"
}

OUTPUT_FILE = os.path.join(os.path.expanduser("~"), "Desktop", "dataset_exploration_results.txt")
log_lines = []

def log(text):
    print(text)
    log_lines.append(text)

# ============================================================
# 3. PTB-XL ANALİZİ
# ============================================================
def analyze_ptbxl(base_path):
    log("\n" + "="*70)
    log("PTB-XL VERİ SETİ ANALİZİ")
    log("="*70)
    
    csv_path = os.path.join(base_path, "ptbxl_database.csv")
    if not os.path.exists(csv_path):
        log(f"HATA: {csv_path} bulunamadı!")
        return
    
    df = pd.read_csv(csv_path, index_col='ecg_id')
    
    log(f"\n--- Temel Bilgiler ---")
    log(f"Toplam kayıt sayısı: {len(df)}")
    log(f"Benzersiz hasta sayısı: {df['patient_id'].nunique()}")
    
    # Yaş
    log(f"\n--- Yaş Dağılımı ---")
    log(f"Ortalama: {df['age'].mean():.1f}, Std: {df['age'].std():.1f}")
    log(f"Min: {df['age'].min():.0f}, Max: {df['age'].max():.0f}, Median: {df['age'].median():.0f}")
    
    # Cinsiyet
    sex_counts = df['sex'].value_counts()
    log(f"\n--- Cinsiyet Dağılımı ---")
    for sex_val, count in sex_counts.items():
        label = "Erkek" if sex_val == 1 else "Kadın"
        log(f"{label} (sex={sex_val}): {count} ({100*count/len(df):.1f}%)")
    
    # Diagnostic superclass
    df['scp_codes'] = df['scp_codes'].apply(lambda x: ast.literal_eval(x))
    scp_path = os.path.join(base_path, "scp_statements.csv")
    agg_df = pd.read_csv(scp_path, index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]
    
    def get_superclass(scp_dict):
        classes = []
        for key in scp_dict.keys():
            if key in agg_df.index:
                classes.append(agg_df.loc[key].diagnostic_class)
        return list(set(classes))
    
    df['superclass'] = df['scp_codes'].apply(get_superclass)
    
    all_classes = []
    for cls_list in df['superclass']:
        all_classes.extend(cls_list)
    class_counts = Counter(all_classes)
    
    log(f"\n--- Diagnostic Superclass Dağılımı (5 sınıf) ---")
    for cls, count in class_counts.most_common():
        log(f"{cls}: {count} ({100*count/len(df):.1f}%)")
    
    # Multi-label
    df['num_classes'] = df['superclass'].apply(len)
    log(f"\n--- Multi-label Durumu ---")
    for n in sorted(df['num_classes'].unique()):
        count = (df['num_classes'] == n).sum()
        log(f"{n} sınıflı kayıtlar: {count} ({100*count/len(df):.1f}%)")
    
    # Sinyal bilgisi
    log(f"\n--- Sinyal Bilgileri ---")
    log(f"Sampling rate: 500 Hz (records500) / 100 Hz (records100)")
    log(f"Kayıt süresi: Tüm kayıtlar 10 saniye (sabit)")
    log(f"Lead sayısı: 12")
    
    # Fold dağılımı
    log(f"\n--- Stratified Fold Dağılımı ---")
    fold_counts = df['strat_fold'].value_counts().sort_index()
    for fold, count in fold_counts.items():
        role = "TEST" if fold == 10 else "TRAIN"
        log(f"Fold {int(fold)}: {count} kayıt [{role}]")
    
    # Device
    log(f"\n--- Cihaz Bilgisi ---")
    device_counts = df['device'].value_counts()
    for dev, count in device_counts.items():
        log(f"'{dev.strip()}': {count}")
    
    return df

# ============================================================
# 4. CHAPMAN & CPSC 2018 ANALİZİ
# ============================================================
def analyze_hea_dataset(base_path, dataset_name):
    log("\n" + "="*70)
    log(f"{dataset_name} VERİ SETİ ANALİZİ")
    log("="*70)
    
    hea_files = glob.glob(os.path.join(base_path, "**", "*.hea"), recursive=True)
    if not hea_files:
        log(f"HATA: {base_path} içinde .hea dosyası bulunamadı!")
        return None
    
    log(f"\nToplam kayıt sayısı: {len(hea_files)}")
    
    ages = []
    sexes = []
    all_dx_codes = []
    dx_per_record = []
    signal_lengths = []
    sampling_rates = []
    errors = 0
    
    for i, hea_file in enumerate(hea_files):
        if (i+1) % 2000 == 0:
            print(f"  ...{i+1}/{len(hea_files)} dosya işlendi")
        try:
            with open(hea_file, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            # İlk satırdan sinyal bilgisi
            first_line = lines[0].strip().split()
            if len(first_line) >= 4:
                fs = float(first_line[2])
                num_samples = int(first_line[3])
                sampling_rates.append(fs)
                signal_lengths.append(num_samples)
            
            for line in lines:
                line = line.strip()
                if line.startswith('#Age:'):
                    age_str = line.replace('#Age:', '').strip()
                    try:
                        age = float(age_str)
                        if 0 < age < 120:
                            ages.append(age)
                    except:
                        pass
                
                elif line.startswith('#Sex:'):
                    sex = line.replace('#Sex:', '').strip()
                    sexes.append(sex)
                
                elif line.startswith('#Dx:') or line.startswith('# Dx:'):
                    dx_str = line.replace('#Dx:', '').replace('# Dx:', '').strip()
                    codes = [c.strip() for c in dx_str.split(',') if c.strip()]
                    all_dx_codes.extend(codes)
                    dx_per_record.append(len(codes))
        except Exception as e:
            errors += 1
    
    if errors > 0:
        log(f"Okunamayan dosya sayısı: {errors}")
    
    # Sinyal uzunlukları
    if signal_lengths:
        lengths_sec = [s / fs for s, fs in zip(signal_lengths, sampling_rates)]
        log(f"\n--- Sinyal Uzunlukları ---")
        
        fs_counts = Counter(sampling_rates)
        log(f"Sampling rate dağılımı: {dict(fs_counts)}")
        log(f"Lead sayısı: 12")
        log(f"Sample sayısı - Min: {min(signal_lengths)}, Max: {max(signal_lengths)}, Ort: {np.mean(signal_lengths):.0f}")
        log(f"Süre (sn) - Min: {min(lengths_sec):.1f}, Max: {max(lengths_sec):.1f}, Ort: {np.mean(lengths_sec):.1f}")
        
        # Süre dağılımı
        log(f"\nSüre dağılımı:")
        ranges = [
            ('<6s', lambda l: l < 6),
            ('6-9.9s', lambda l: 6 <= l < 9.9),
            ('10s', lambda l: 9.9 <= l <= 10.1),
            ('10.1-15s', lambda l: 10.1 < l <= 15),
            ('15-30s', lambda l: 15 < l <= 30),
            ('30-60s', lambda l: 30 < l <= 60),
            ('>60s', lambda l: l > 60),
        ]
        for name, cond in ranges:
            count = sum(1 for l in lengths_sec if cond(l))
            if count > 0:
                log(f"  {name}: {count} kayıt ({100*count/len(lengths_sec):.1f}%)")
    
    # Yaş
    if ages:
        log(f"\n--- Yaş Dağılımı ---")
        log(f"Geçerli yaş verisi: {len(ages)} / {len(hea_files)}")
        log(f"Ortalama: {np.mean(ages):.1f}, Std: {np.std(ages):.1f}")
        log(f"Min: {min(ages):.0f}, Max: {max(ages):.0f}, Median: {np.median(ages):.0f}")
    
    # Cinsiyet
    if sexes:
        sex_counts = Counter(sexes)
        log(f"\n--- Cinsiyet Dağılımı ---")
        for sex, count in sex_counts.most_common():
            log(f"{sex}: {count} ({100*count/len(hea_files):.1f}%)")
    
    # SNOMED kodları
    code_counts = Counter(all_dx_codes)
    log(f"\n--- SNOMED-CT Tanı Kodları ---")
    log(f"Toplam benzersiz kod sayısı: {len(code_counts)}")
    log(f"\nTüm kodlar ve frekansları:")
    for code, count in code_counts.most_common():
        log(f"  {code}: {count} kayıt ({100*count/len(hea_files):.1f}%)")
    
    # Multi-label
    if dx_per_record:
        log(f"\n--- Multi-label Durumu ---")
        dx_counter = Counter(dx_per_record)
        for n in sorted(dx_counter.keys()):
            count = dx_counter[n]
            log(f"{n} tanılı kayıtlar: {count} ({100*count/len(hea_files):.1f}%)")
    
    return code_counts

# ============================================================
# 5. ÇALIŞTIR
# ============================================================
if __name__ == "__main__":
    log("ECG DATASET EXPLORATION REPORT")
    log(f"Tarih: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    log("="*70)
    
    for name, path in PATHS.items():
        exists = os.path.exists(path)
        log(f"{name}: {'BULUNDU' if exists else 'BULUNAMADI!'} --> {path}")
    
    # PTB-XL
    ptbxl_df = analyze_ptbxl(PATHS['PTB-XL'])
    
    # Chapman
    log("\n(Chapman taranıyor, birkaç dakika sürebilir...)")
    chapman_codes = analyze_hea_dataset(PATHS['Chapman'], 'CHAPMAN-SHAOXING')
    
    # CPSC 2018
    log("\n(CPSC 2018 taranıyor...)")
    cpsc_codes = analyze_hea_dataset(PATHS['CPSC2018'], 'CPSC 2018')
    
    # Karşılaştırmalı özet
    log("\n" + "="*70)
    log("KARŞILAŞTIRMALI ÖZET")
    log("="*70)
    
    if chapman_codes and cpsc_codes:
        chapman_set = set(chapman_codes.keys())
        cpsc_set = set(cpsc_codes.keys())
        common = chapman_set & cpsc_set
        only_chapman = chapman_set - cpsc_set
        only_cpsc = cpsc_set - chapman_set
        log(f"Chapman benzersiz kod: {len(chapman_set)}")
        log(f"CPSC 2018 benzersiz kod: {len(cpsc_set)}")
        log(f"Ortak kodlar ({len(common)}): {common}")
        log(f"Sadece Chapman ({len(only_chapman)}): {only_chapman}")
        log(f"Sadece CPSC ({len(only_cpsc)}): {only_cpsc}")
    
    # Kaydet
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(log_lines))
    
    log(f"\nSonuçlar kaydedildi: {OUTPUT_FILE}")
    log("Bu dosyayı Claude'a gönder!")
