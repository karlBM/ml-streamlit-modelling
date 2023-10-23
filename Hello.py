# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import streamlit as st
from streamlit.logger import get_logger

LOGGER = get_logger(__name__)


def run():
    st.set_page_config(
        page_title="Hello",
        page_icon="👋",
    )

    st.write("# Welcome to KarlAI! 👋")

    st.sidebar.success("Select a model above.")

    st.markdown(
        """
        This is a demo app where you can build, evaluate and deploy
        ml models. \n
        **👈 Select a model from the sidebar** to see some examples
        of what ml can do!
        ### Want to learn more?
        - Check out this webpage ** coming :p 
        - Jump into our [documentation] --> DS documentation
        - Ask a question in our DS jira
        ### See more complex demos
        - pricing and forecasting 
        - Some cool GEO location example
    """
    )


if __name__ == "__main__":
    run()
