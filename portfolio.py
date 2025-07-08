import chromadb
import uuid
import pandas as pd







class Portfolio:
    def __init__(self,file_path="resource/my_portfolio.csv"):
        self.file_path=file_path
        self.df=pd.read_csv(file_path)

        self.client= chromadb.Client("vectorstore")
        self.collection=self.client.get_or_create_collection(name="portfolio")
    

    def load_portfolio(self):

        if not self.collection.count():
            for _,row in self.df.iterrows():
                self.collection.add(documents=row["Techstack"],
                metadatas={"links": row["Links"]},
                ids=[str(uuid.uuid4())])
    
    def query_links(self,skills):
        return self.collection.query(query_texts=skills,n_results=2).get("metadatas",[])
    