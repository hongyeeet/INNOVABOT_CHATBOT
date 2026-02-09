# INNOVABOT Chatbot – Setup Guide


```bash
cd INNOVABOT_PROJECT

python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# or Command Prompt
.venv\Scripts\activate.bat


pip install --upgrade pip
pip install -r requirements.txt


#DO NOT RUN THIS (ALERADY RAN) - only run if brochure_bm25.pkl , brochure_chunks.pkl or brochure_faiss.index are deleted or corrupted 
python build_brochure_index.py 

# open .env file and enter open AI api key
OPENAI_API_KEY=your_openai_key_here

# Run in terminal 
streamlit run main.py
