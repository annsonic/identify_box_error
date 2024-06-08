from math import ceil

import streamlit as st


def menu():
    """ Show a navigation menu for plots. """
    with st.sidebar:
        st.title("Which kind of analysis would you like to see?")
        proportion = st.button(label="Error Proportions", use_container_width=True)
        errors_per_image = st.button(label="Errors per Image", use_container_width=True)
        impact = st.button(label="Error Impact on mAP", use_container_width=True)

        if proportion:
            st.switch_page("pages/proportions.py")
        if errors_per_image:
            st.switch_page("pages/errors_per_image.py")
        if impact:
            st.switch_page("pages/impact.py")


def target_classes_form():
    class_names = st.session_state.class_names
    num_cols = 5
    num_rows = ceil(len(class_names) / num_cols)
    targets = []

    with st.form('target_classes_form', clear_on_submit=True):
        st.subheader('Select the classes you want to analyze:')
        for i in range(num_rows):
            cols = st.columns(num_cols)
            for j in range(num_cols):
                if i * num_cols + j >= len(class_names):
                    break
                with cols[j]:
                    targets.append(st.checkbox(class_names[i * num_cols + j], value=False))
        cols = st.columns(2)
        with cols[0]:
            analyze = st.form_submit_button('Analyze', type='primary')
        with cols[1]:
            reset = st.form_submit_button('Reset', type='primary')

        if analyze:
            target_classes = [i for i, selected in enumerate(targets) if selected]
            st.session_state['target_classes'] = target_classes
        if reset:
            st.session_state['target_classes'] = []
            st.rerun()  # Rerun the script
