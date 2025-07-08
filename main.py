import streamlit as st
from langchain_community.document_loaders import WebBaseLoader

from chains import Chain
from portfolio import Portfolio
from utils import clean_text

def create_streamlit_app(llm,portfolio,clean_text):


    st.title ("🤖🤖Cold mail generator🤖🤖")
    url_input=st.text_input("Enter job posting url",value="")
    gen_button=st.button("Generate")

    if gen_button:
        try:           

            loader = WebBaseLoader([url_input])
            page_data=clean_text(loader.load().pop().page_content)

            
            portfolio.load_portfolio()
            jobs=llm.extract_jobs(page_data)
            for job in jobs:
                skills=job.get("skills",[])
                links=portfolio.query_links(skills)
                email=llm.write_email(job,links)
                st.code(email,language="markdown")
        except Exception as e:
            st.error(f"Error occured {e}")


if __name__=="__main__":
    chain=Chain()
    portfolio=Portfolio()
    st.set_page_config(layout="wide",page_title="🤖🤖Cold mail generator🤖🤖",page_icon="📧📧")
    create_streamlit_app(chain,portfolio,clean_text)
