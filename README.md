# 🩸 Blood Work Visual Analyzer

An intelligent health dashboard that extracts lab values from unstructured blood report text using LLMs, standardizes clinical data into normalized Plotly visual charts, and generates culturally specific (Sri Lankan) dietary recommendations with automatic multi-provider LLM fallback.

## ✨ Key Features

* **Automated Data Extraction:** Converts raw blood report text into structured JSON (test names, values, units, reference ranges, status) using GenAI.
* **Normalized Visual Analytics:** Plots test markers on a relative **0% – 100% scale** (0% = Min Reference Limit, 100% = Max Reference Limit) to easily visualize out-of-range markers regardless of unit scale differences.
* **Multi-Provider Fallback Architecture:** Automatically fails over from **Google Gemini 1.5 Flash** to **OpenAI GPT-4o-Mini** (or vice versa) if rate limits or API errors occur.
* **Localized Dietary Insights:** Provides targeted clinical health summaries and practical Sri Lankan diet recommendations based on detected abnormal lab results.

## 🛠️ Tech Stack

* **Frontend:** Streamlit
* **Data Processing:** Pandas
* **Data Visualization:** Plotly Express / Plotly Graph Objects
* **LLM Orchestration:** LangChain (`langchain-google-genai`, `langchain-openai`)
* **LLM Models:** Google Gemini (`gemini-1.5-flash`), OpenAI (`gpt-4o-mini`)

## 🚀 Getting Started

### 1. Clone the Repository
git clone [https://github.com/your-username/blood-report-analysis.git](https://github.com/your-username/blood-report-analysis.git)
cd blood-report-analysis

### 2. Environment Setup
Create a .env file in the root directory and add your API keys:
GOOGLE_API_KEY=your_google_gemini_api_key
OPENAI_API_KEY=your_openai_api_key

### 3. Install Dependencies
Using uv:
uv sync

### 4. Run the Application
python -m streamlit run src/blood_report_analysis/ui/app.py
