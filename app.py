# @title 기본 UI 및 모델 로딩

import streamlit as st
from transformers import pipeline
from PIL import Image

# 1. 페이지 기본 설정
st.set_page_config(
    page_title="이미지 분류",
    page_icon="🖼️"
)


# 2. UI 레이아웃 구성
st.title("이미지 분류")
st.markdown("---")
st.write("이미지를 업로드하면 AI가 분석하여 어떤 대상인지 알려줍니다.")


# 3. 모델 로딩 함수
@st.cache_resource
def load_model():
    model = pipeline("image-classification", model="google/vit-base-patch16-224")
    return model

# 4. 모델 로드 실행
with st.spinner("AI 모델을 다운로드 및 로딩합니다"):
    classifier = load_model()

st.success("모델 준비완료")

# 5. 파일 업로더 생성
uploaded_file = st.file_uploader("분석할 이미지를 올려주세요", type=["jpg","png","jpeg"])

# 6. 이미지 처리 및 추론
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='업로드된 이미지', use_container_width=True)

    # 분석 버튼
    if st.button("이미지 분석 실행"):
        with st.spinner("이미지 분석중..."):
            predictions = classifier(image)

            # 높은 확률의 겨로가 가져오기
            top_prediction = predictions[0]
            label = top_prediction["label"]
            score = top_prediction["score"]

            st.markdown("---")
            st.subheader("분석 결과")

            # metric으로 결과 강조 표시
            st.metric(label="예측된 대상", value=label, delta=f"{score * 100:.1f}% 확신")

            # 확률 시각화
            st.write("신뢰도 :")
            st.progress(score)

            # 상위 5개 결과보여주기
            with st.expander("다른 후보들 보기"):
                for pred in predictions:
                    st.write(f"{pred['label']}: {pred['score']*100:.1f}%")