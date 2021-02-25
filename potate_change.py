import streamlit as st
import numpy as np
from PIL import Image
import cv2
from mtcnn.mtcnn import MTCNN
import random


def face_detect_MTCNN(img):
    b_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    detect = MTCNN()
    faces = detect.detect_faces(b_img)
    list = [0, 0, 0, 0, 0, 0, 1]
    for i in range(len(faces)):
        (x, y, w, h) = faces[i]["box"]
        num = random.choice(list)
        if num == 0:
            persona = cv2.imread("potato_boypersona.jpg")
        else:
            persona = cv2.imread("koikeya.jpg")
        persona = cv2.resize(persona, (w, int(h * 0.8)))
        persona = cv2.cvtColor(persona, cv2.COLOR_BGR2RGB)
        # img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        img[y:y + int(h * 0.8), x:x + w] = persona

    return img

def scale_to_width(img,width):  # PIL画像をアスペクト比を固定してリサイズする。
    im_height = img.height
    im_width = img.width
    aspect = im_height / im_width
    if im_height > im_width:
        return img.resize((width, int(width * aspect)))
    else:
        return img.resize((int(width / aspect), width))


if __name__ == '__main__':
    st.title("みんなポテト坊やだよ！！！")
    """
    **~Let`s CHANGE POTATE-BOY!!!~**
    """
    uploaded_file = st.file_uploader("ここから誰かの写真を撮ってね！（横長）", type=["png", "jpg"], accept_multiple_files=False)
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image = scale_to_width(image,1000)#リサイズ
        image = np.array(image.convert("RGB"))
        image = cv2.cvtColor(image, 1)

        st.image(face_detect_MTCNN(image))
        # image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    uploaded_file_h = st.file_uploader("縦撮影はこちらから", type=["png", "jpg"], accept_multiple_files=False)
    if uploaded_file_h is not None:
        image = Image.open(uploaded_file_h)
        image = scale_to_width(image, 1000)  # リサイズ
        image = np.array(image.convert("RGB"))
        image = cv2.cvtColor(image, 1)
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

        st.image(face_detect_MTCNN(image))

