import cv2, sys
sys.path.insert(0,'/app')
from pathlib import Path
from app.services.jersey_ocr import read_jersey_number_bgr, jersey_ocr_enabled, _pick_best_number, prefer_jersey_number
import numpy as np
print('enabled', jersey_ocr_enabled())
print('pick 13', _pick_best_number([(1,80),(13,70)]))
print('vote 23', prefer_jersey_number([(2,0.9),(23,0.7)]))
for text in ['13','23','10','7','11']:
    img = np.full((160,120,3),(40,40,200),np.uint8)
    cv2.putText(img,text,(20,105),cv2.FONT_HERSHEY_SIMPLEX,2.2,(255,255,255),5)
    hit = read_jersey_number_bgr(img)
    print(f' synthetic {text} -> {hit}')
root = Path('/host_data/kaggle_out/corpus_v26/corpus_out')
for slug in ['spain_france_u19_2023','portugal_spain_euro_2026_final','test_video_2']:
    print('==', slug)
    for p in sorted((root/slug/'thumbs').glob('*.jpg')):
        hit = read_jersey_number_bgr(cv2.imread(str(p)))
        if hit:
            print(f'  {p.name}: {hit}')
