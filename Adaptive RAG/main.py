from dotenv import load_dotenv
load_dotenv()
from graph.graph import app
from langchain_core.runnables.graph_mermaid import draw_mermaid_png, MermaidDrawMethod
from pyppeteer import launch

if __name__ == "__main__":
    print("Starting Advanced RAG...")
    print(app.invoke(input={"question": "Whats is agent memory?"}))
    
    
