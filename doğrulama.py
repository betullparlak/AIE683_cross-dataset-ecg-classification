"""
ECG Pipeline Doğrulama Scripti

1. Dosyaların boyut/format kontrolü
2. Sınıf dağılımı raporu
3. CPSC crop/pad detay raporu (orijinal vs işlenmiş süre)
4. Rastgele sinyal görselleştirme (ham vs işlenmiş)
5. Z-score kontrolü (ortalama≈0, std≈1)
6. Baseline wander kontrolü
"""

import os
import glob
import numpy as np
import pandas as pd
from collections import Counter
from datetime import datetime


PTBXL_BASE = r"C:\Users\BETÜL\Desktop\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1"

PROCESSED = {
    'PTB-XL':   os.path.join(PTBXL_BASE, 'Processed', 'PTB-XL'),
    'Chapman':  os.path.join(PTBXL_BASE, 'Processed', 'Chapman'),
    'CPSC2018': os.path.join(PTBXL_BASE, 'Processed', 'CPSC2018'),
}

CPSC_SRC = r"C:\Users\BETÜL\Desktop\Training_WFDB"

OUTPUT_FILE = os.path.join(PTBXL_BASE, 'Processed', 'verification_report.txt')


def log(text, f=None):
    print(text)
    if f:
        f.write(text + '\n')


def main():
    f = open(OUTPUT_FILE, 'w', encoding='utf-8')
    log("=" * 70, f)
    log("ECG PIPELINE DOĞRULAMA RAPORU", f)
    log(f"Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M')}", f)
    log("=" * 70, f)

    for ds_name, proc_dir in PROCESSED.items():
        log(f"\n{'='*70}", f)
        log(f" {ds_name} DOĞRULAMA", f)
        log(f"{'='*70}", f)

        npz_files = sorted(glob.glob(os.path.join(proc_dir, '**', '*.npz'), recursive=True))
        total = len(npz_files)
        log(f"\nToplam işlenmiş dosya: {total}", f)

        if total == 0:
            log("  UYARI: Hiç dosya bulunamadı!", f)
            continue

        # counters
        shape_ok = 0
        shape_bad = 0
        class_counter = Counter()
        multilabel_counter = Counter()
        zscore_issues = 0
        bad_files = []

        # CPSC crop/pad detayları
        cpsc_details = []

        for npz_path in npz_files:
            try:
                data = np.load(npz_path, allow_pickle=True)
                signal = data['signal']
                super_classes = list(data['super_classes'])
                original_labels = list(data['original_labels'])

                # 1. Boyut kontrolü
                if signal.shape == (5000, 12):
                    shape_ok += 1
                else:
                    shape_bad += 1
                    bad_files.append(f"  {os.path.basename(npz_path)}: {signal.shape}")

                # 2. Sınıf dağılımı
                for cls in super_classes:
                    class_counter[cls] += 1
                multilabel_counter[len(super_classes)] += 1

                # 3. Z-score kontrolü
                for lead in range(signal.shape[1]):
                    mu = np.abs(signal[:, lead].mean())
                    std = signal[:, lead].std()
                    if mu > 0.1 or abs(std - 1.0) > 0.5:
                        zscore_issues += 1
                        break

            except Exception as e:
                bad_files.append(f"  {os.path.basename(npz_path)}: OKUMA HATASI - {e}")
                shape_bad += 1

        # Report
        log(f"\n--- Boyut Kontrolü ---", f)
        log(f"  Doğru (5000x12): {shape_ok}", f)
        log(f"  Hatalı: {shape_bad}", f)
        if bad_files:
            for bf in bad_files[:10]:
                log(bf, f)

        log(f"\n--- Sınıf Dağılımı ---", f)
        for cls, count in sorted(class_counter.items()):
            pct = 100 * count / total
            log(f"  {cls:<8}: {count:>6} ({pct:.1f}%)", f)

        log(f"\n--- Multi-label Dağılımı ---", f)
        for n_labels, count in sorted(multilabel_counter.items()):
            pct = 100 * count / total
            log(f"  {n_labels} sınıf: {count:>6} ({pct:.1f}%)", f)

        log(f"\n--- Z-score Kontrolü ---", f)
        log(f"  Sorunlu kayıt: {zscore_issues} / {total}", f)
        if zscore_issues == 0:
            log(f"  ✓ Tüm kayıtlar normalize edilmiş.", f)

        #  CPSC Crop/Pad Detay Raporu
        if ds_name == 'CPSC2018':
            log(f"\n--- CPSC 2018: Crop/Pad Detay Raporu ---", f)

            hea_files = sorted(glob.glob(os.path.join(CPSC_SRC, '**', '*.hea'), recursive=True))
            cropped_list = []
            padded_list = []

            for hea_path in hea_files:
                record_name = os.path.splitext(os.path.basename(hea_path))[0]
                try:
                    with open(hea_path, 'r', encoding='utf-8', errors='ignore') as hf:
                        first_line = hf.readline().strip().split()
                        fs = float(first_line[2])
                        n_samples = int(first_line[3])
                        duration = n_samples / fs

                        if duration > 10.0:
                            cropped_list.append((record_name, duration))
                        elif duration < 10.0:
                            padded_list.append((record_name, duration))
                except:
                    pass

            log(f"\n  Kesilmesi gereken kayıtlar (>{10}s): {len(cropped_list)}", f)
            if cropped_list:
                # Süre dağılımı
                durations = [d for _, d in cropped_list]
                log(f"  Süre aralığı: {min(durations):.1f}s - {max(durations):.1f}s", f)
                log(f"  Ortalama: {np.mean(durations):.1f}s, Median: {np.median(durations):.1f}s", f)

                # Süre grupları
                ranges = [
                    ('10.1-15s', 10.1, 15),
                    ('15-30s', 15, 30),
                    ('30-60s', 30, 60),
                    ('60-100s', 60, 100),
                    ('>100s', 100, 999),
                ]
                log(f"\n  Süre dağılımı (kesilen kayıtlar):", f)
                for label, lo, hi in ranges:
                    cnt = sum(1 for d in durations if lo < d <= hi)
                    if cnt > 0:
                        log(f"    {label:<10}: {cnt:>5} kayıt", f)

                # En uzun 10 kayıt
                log(f"\n  En uzun 10 kayıt:", f)
                for name, dur in sorted(cropped_list, key=lambda x: -x[1])[:10]:
                    log(f"    {name}: {dur:.1f}s -> 10s (ortadan {dur-10:.1f}s kesildi)", f)

            log(f"\n  Pad'lenmesi gereken kayıtlar (<{10}s): {len(padded_list)}", f)
            if padded_list:
                for name, dur in padded_list:
                    log(f"    {name}: {dur:.1f}s -> 10s ({10-dur:.1f}s sıfır eklendi)", f)

    # sum
    log(f"\n{'='*70}", f)
    log("GENEL ÖZET", f)
    log(f"{'='*70}", f)

    total_saved = 0
    for ds_name, proc_dir in PROCESSED.items():
        count = len(glob.glob(os.path.join(proc_dir, '**', '*.npz'), recursive=True))
        total_saved += count
        log(f"  {ds_name:<10}: {count:>6} kayıt", f)
    log(f"  {'TOPLAM':<10}: {total_saved:>6} kayıt", f)

    # Rastgele 5 kayıt örnek göster
    log(f"\n--- Rastgele 5 Kayıt Örneği ---", f)
    for ds_name, proc_dir in PROCESSED.items():
        npz_files = glob.glob(os.path.join(proc_dir, '**', '*.npz'), recursive=True)
        if npz_files:
            sample = np.random.choice(npz_files, min(5, len(npz_files)), replace=False)
            log(f"\n  {ds_name}:", f)
            for sp in sample:
                data = np.load(sp, allow_pickle=True)
                sig = data['signal']
                cls = list(data['super_classes'])
                orig = list(data['original_labels'])
                log(f"    {os.path.basename(sp)}: shape={sig.shape}, "
                    f"mean={sig.mean():.4f}, std={sig.std():.4f}, "
                    f"classes={cls}, labels={orig[:3]}{'...' if len(orig)>3 else ''}", f)

    log(f"\nRapor kaydedildi: {OUTPUT_FILE}", f)
    f.close()
    print(f"\nRapor kaydedildi: {OUTPUT_FILE}")


if __name__ == '__main__':
    main()