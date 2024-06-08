from pathlib import Path

import numpy as np
import streamlit as st

from app.task.study_in_statistics import recorder, Statistician
from menu import menu, target_classes_form


@st.experimental_dialog("Visualize the image with error type", width='large')
def display_annotation_image(anno_file_name: str):
    left, right = st.columns([0.25, 1])
    left.image(str(Path(st.session_state.analysis_folder_path) / "images" / "color_legend.png"))
    left.subheader(anno_file_name)
    right.image(str(Path(st.session_state.analysis_folder_path) / "images" / anno_file_name))


menu()
st.title("Errors per Image")
target_classes_form()

if st.session_state.analysis_folder_path:
    file_name = 'error_type_histogram.png'
    if st.session_state.target_classes:
        file_name = f"class({'_'.join(map(str, st.session_state.target_classes))})_{file_name}"
    file_path = st.session_state.analysis_folder_path / 'errors_per_image' / file_name
    # Always recompute the histogram to get the dataframe
    billboard = Statistician(
        output_folder_path=st.session_state.analysis_folder_path,
        df=recorder.read_csv(st.session_state.analysis_folder_path / f"{st.session_state.data_subset}.csv")
    ).sort_by_errors_per_image(target_classes=st.session_state.target_classes)
    class_names = [st.session_state.class_names[i] for i in st.session_state.target_classes]

    with st.container(border=True, height=540):
        col1, col2 = st.columns(2)
        col1.subheader("You have chosen the following options:")
        info = f"- Analyze the split dataset:\n  - {st.session_state.data_subset}\n"
        info += "- Target object classes:\n"
        for name in class_names:
            info += f"  - {name}\n"
        if not class_names:
            info += "  - All classes\n"
        col1.write(info)
        col2.image(str(file_path))
    
    # col1, col2 = st.columns(2)
    # with col1:
        # st.subheader("The error rankings:")
        # st.dataframe(billboard)
    # with col2:
    img_names = [""]
    for img_name in billboard.index:
        count = billboard['error_count'][img_name]
        img_names.append(img_name + f" ({count})")
    with st.container(border=True):
        st.subheader("You can select the images to visualize:")
        # a_file_name = st.selectbox("Select the image file", [""] + billboard.index.to_list())
        a_file_name = st.selectbox("Options are `image file name (error counts)`", img_names, label_visibility="visible")
        if a_file_name:
            tokens = a_file_name.split()
            a_file_name = ''.join(tokens[:-1])
            display_annotation_image(a_file_name)
    