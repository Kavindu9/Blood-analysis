from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-3.7-flash")

with open("../../blood_Report_Sample-1_page.txt", "r") as f:
    blood_report = f.read()

print(blood_report[:400])


extraction_prompt = f"""
You are a medical data extraction assistant.

From the blood report below, extract all text values and classify each one as HIGH, LOW, or NORMAL based on the refernce ranges provided in the report

Format your response as:
- Test Name: value | Status: HIGH?LOW/NORMAL | Reference: range

Blood Report:
{blood_report}
"""

extraction_data = llm.invoke(extraction_prompt)
extract_values = extraction_data.text

print("=== STAGE 1: Extracted Values ===")
print(extract_values)




diet_prompt = f"""
You are a clinical nutritionist specializin in Sri Lankan dietary habits.

Based on the blood work analysis below, write:
    1. A shoet helath summary in 4-5 lines explaining the patient's condition in simple language
    2. A short, practicle Sri Lankan diet plan having only two sections (1) Foods to avoid (2) Foods to eat more of.

Blood Work Analysis:
{extraction_values}
"""

diet = llm.invoke(diet_prompt)

print("=== STAGE 2: Health Summary and Diet Plan ===")
print(diet.text)
