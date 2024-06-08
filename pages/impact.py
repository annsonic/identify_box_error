from pathlib import Path

import streamlit as st

from app.task.study_in_statistics import recorder, Statistician
from menu import menu, target_classes_form


menu()
st.title("Impact on mAP")
target_classes_form()

if st.session_state.analysis_folder_path:
    file_name = 'delta_map.png'
    if st.session_state.target_classes:
        file_name = f"class({'_'.join(map(str, st.session_state.target_classes))})_{file_name}"
    file_path = st.session_state.analysis_folder_path / 'impact' / file_name
    if not file_path.exists():
        size = len(st.session_state['class_names']) if not st.session_state.target_classes else None
        Statistician(
            output_folder_path=st.session_state.analysis_folder_path,
            df=recorder.read_csv(st.session_state.analysis_folder_path / f"{st.session_state.data_subset}.csv")
        ).delta_map(target_classes=st.session_state.target_classes, 
                    num_classes=size)
    class_names = [st.session_state.class_names[i] for i in st.session_state.target_classes]

    with st.container(border=True, height=240):
        st.subheader("You have chosen the following options:")
        info = f"- Analyze the split dataset:\n  - {st.session_state.data_subset}\n"
        info += "- Target object classes:\n"
        for name in class_names:
            info += f"  - {name}\n"
        if not class_names:
            info += "  - All classes\n"
        st.write(info)
    
    st.image(str(file_path))