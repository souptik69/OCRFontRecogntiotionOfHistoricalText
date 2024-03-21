import sys, os
wd = os.getcwd()
sys.path.append(os.path.split(wd)[0])
from typing import List

import streamlit as st


class Visualizer():
    def __init__(self):
        if "current_page_number" not in st.session_state:
            st.session_state["current_page_number"] = 0
        self.entries = 394

        self.sidebar()
        st.text("Current page: " + str(st.session_state["current_page_number"]))


    def sidebar(self):
        st.sidebar.number_input("input #", key="current_page_number")

        # if st.session_state["current_page_number"] is not 0:
        #     if st.sidebar.button("Previous page"):
        #         st.session_state["current_page_number"] -= 1
        # else:
        #     st.sidebar.text("Start!")
        #
        # if st.session_state["current_page_number"] is not (self.entries - 1):
        #     if st.sidebar.button("Next page"):
        #         st.session_state["current_page_number"] += 1
        # else:
        #     st.sidebar.text("Done!")



if __name__ == '__main__':
    sys.path.append("..")
    st.set_page_config(layout='wide', page_title="NBB Recipient Labeling")
    vis = Visualizer()