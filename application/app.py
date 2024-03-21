import sys, os
wd = os.getcwd()
sys.path.append(os.path.split(wd)[0])
from typing import List

import streamlit as st

from helper.database_helper import connect, toggle_line_info
from helper.pageEntry import PageEntry
from helper.page_helper import load_page_and_draw, build_entries


class Visualizer:
    def __init__(self):

        self.connection = self.get_database_connection()
        st.title("Working package 3: Addressee detection")
        self.books = ["Band6"]
        # self.books = ["Band6", "Band7", "Band8", "Band9"]
        # self.book = "Band6"

        if "current_page_number" not in st.session_state:
            st.session_state["current_page_number"] = 0
        if "current_book" not in st.session_state:
            st.session_state["current_book"] = "Band6"

        self.data_path = os.path.join("tmp", "new_dataset")
        self.entries: List[PageEntry] = self.build_entries_cached(self.data_path, st.session_state["current_book"])
        print("found {} entries".format(len(self.entries)))

        self.sidebar()
        page_image = self._load_entry()
        self.main_fragment(page_image)

    def _load_entry(self):
        self.current_page: PageEntry = self.entries[st.session_state["current_page_number"]]
        self.current_page.compute(visualization=True, connection=self.connection)

        page_recipient_ids = [line.id for line in self.current_page.page_line_entries if line.is_recipient]
        page_image = self.load_page_and_draw_cached(self.current_page, page_recipient_ids)
        return page_image

    def sidebar(self):
        st.sidebar.title("Settings")
        self.book = st.sidebar.selectbox(label="Choose book", options=self.books, key="current_book")

        st.sidebar.number_input("Jump to page number (max: {}.)".format((len(self.entries)-1)),
                                                                          key="current_page_number",
                                                                          min_value=0,
                                                                          max_value=(len(self.entries)-1))


    def main_fragment(self, page_image):
        main_col0, main_col1, main_col2 = st.columns((1, 1, 1))
        main_col0.image(page_image, caption=self.current_page.image_file)
        main_col1.header(" ")

        for i, line in enumerate(self.current_page.page_line_entries):
            # hack to be able to call on_change method for checkbox
            col_to_print = main_col1 if line.line_number < 20 else main_col2

            checkbox_text = f"Recipient info for line {line.line_number}"
            checkbox_help_text = f"Internal ID: {line.id}"

            col_to_print.checkbox(checkbox_text, value=line.is_recipient,
                                  help=checkbox_help_text, on_change=self.toggle, kwargs={"line": line})


    def toggle(self, line):
        toggle_line_info(self.connection, line)

    @staticmethod
    @st.cache(allow_output_mutation=True)
    def build_entries_cached(data_path, book):
        return build_entries(data_path, book)

    @staticmethod
    @st.cache(suppress_st_warning=True)
    def load_page_and_draw_cached(page_entry: PageEntry, page_recipient_ids):
        return load_page_and_draw(page_entry, page_recipient_ids)

    @staticmethod
    @st.cache(allow_output_mutation=True)
    def get_database_connection():
        return connect()


if __name__ == '__main__':
    sys.path.append("..")
    st.set_page_config(layout='wide', page_title="NBB Recipient Labeling")
    vis = Visualizer()
