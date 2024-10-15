import os
import glob
from tqdm import tqdm

from langchain_community.document_loaders import arxiv, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_community.tools.tavily_search.tool import TavilySearchResults

from langchain import hub
from langchain_openai import OpenAIEmbeddings
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.chains import RetrievalQA
from pydantic.v1 import BaseModel, Field

from utils import get_llm

class DocumentInput(BaseModel):
    question: str = Field()


class IdeaGenerator:
    def __init__(self, config, **kwargs):

        self.llm_model = config['llm_model']
        self.tempoerature = config['llm_temperature']
        self.task_name = config['task_name']
        self.paper_path = os.path.join(config['offline_paper_path'], self.task_name)

        self.llm = get_llm(config)
        self.tools = self.make_tools()
        self.prompt = hub.pull("hwchase17/react")
        agent = create_react_agent(llm=self.llm, tools=self.tools, prompt=self.prompt)
        self.agent_executor = AgentExecutor(agent=agent, tools=self.tools, verbose=False, handle_parsing_errors=True)

    def run(self, user_prompt=None):
        outputs = self.agent_executor.invoke({"input": user_prompt})
        return outputs["output"]

    def make_tools(self):
        offline_papers = self.load_offline_document(paper_path=self.paper_path)
        ## add online search tools
        search = TavilySearchAPIWrapper()
        tavily_tool = TavilySearchResults(
            api_wrapper=search, 
            name="Online Search",
            description="Search online for professional information about the topic.")

        all_tools = offline_papers + [tavily_tool]
        return all_tools

    def load_offline_document(self, paper_path=None):
        if paper_path is None:
            print("No offline paper path provided, online search only")
            return []

        retriever_tools = []
        for pdf_file in glob.glob(paper_path + "/*.pdf"):
            print(f"Loading offline literature: {pdf_file}")
            file_name = pdf_file.split('\\')[-1].split('.')[0]
            loader = PyPDFLoader(pdf_file)
            pages = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            docs = text_splitter.split_documents(pages)
            embeddings = OpenAIEmbeddings()
            retriever = FAISS.from_documents(docs, embeddings).as_retriever()
            
            retriever_tools.append(
                Tool(
                    args_schema=DocumentInput,
                    name="Offline Literature",
                    description=f"Useful when you try to find professional information about the paper: [{file_name}].",
                    func=RetrievalQA.from_chain_type(
                        llm=self.llm, 
                        retriever=retriever, 
                        return_source_documents=True
                    ),
                )
            )
        return retriever_tools


if __name__ == "__main__":
    import os
    from config import configs
    from utils import get_prompt

    os.environ['OPENAI_API_KEY'] = configs['openai_api_key']
    os.environ['tavily_API_KEY'] = configs['tavily_api_key']

    prompt = get_prompt(configs['task_name'])
    idea_generator = IdeaGenerator(config=configs)
    output = idea_generator.run(prompt.idea_generator_prompt)
    print(output)