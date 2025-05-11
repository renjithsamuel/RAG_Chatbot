### Steps to run the application
1. python -m venv venv
2. venv\Scripts\activate
3. pip install -r requirements.txt
4. Put your .txt files in data/docs/
5. Build the index: `python scripts/build_index.py`
6. uvicorn app.main:app --reload
7. test the chatbot: `curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d '{"question": "Explain this doc to me"}'`
