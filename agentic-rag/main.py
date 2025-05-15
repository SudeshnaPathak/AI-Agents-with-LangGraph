from dotenv import load_dotenv
load_dotenv()
from graph.graph import app


if __name__ == "__main__":
    print("Starting Advanced RAG...")
    print(app.invoke(input={"question": "Whats is agent memory?"}))
