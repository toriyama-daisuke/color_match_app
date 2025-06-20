import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

st.set_page_config(page_title="色味合わせアプリ", layout="centered")
st.title("🎨 画像の色味を合わせるアプリ")
st.write("お手本画像と対象画像をアップロードしてください。対象画像の色味が自動で調整されます。")

# ファイルアップロード
source_file = st.file_uploader("お手本画像（source）", type=["jpg", "png", "jpeg"])
target_file = st.file_uploader("対象画像（target）", type=["jpg", "png", "jpeg"])

def match_color(source_img, target_img):
    source = source_img.astype(np.float32)
    target = target_img.astype(np.float32)

    src_mean, src_std = cv2.meanStdDev(source)
    tgt_mean, tgt_std = cv2.meanStdDev(target)

    result = (target - tgt_mean) * (src_std / (tgt_std + 1e-6)) + src_mean
    result = np.clip(result, 0, 255)
    return result.astype(np.uint8)

# 処理
if source_file and target_file:
    source = Image.open(source_file).convert("RGB")
    target = Image.open(target_file).convert("RGB")

    source_np = np.array(source)[:, :, ::-1]  # RGB → BGR
    target_np = np.array(target)[:, :, ::-1]

    target_np_resized = cv2.resize(target_np, (source_np.shape[1], source_np.shape[0]))
    result_np = match_color(source_np, target_np_resized)

    result_pil = Image.fromarray(result_np[:, :, ::-1])  # BGR → RGB

    st.image(result_pil, caption="✅ 色味を合わせた画像", use_column_width=True)

    buf = io.BytesIO()
    result_pil.save(buf, format="JPEG")
    st.download_button("📥 ダウンロード", data=buf.getvalue(), file_name="matched_image.jpg", mime="image/jpeg")
