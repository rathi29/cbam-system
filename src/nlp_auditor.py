import requests
import json
import re

OLLAMA_URL = "http://localhost:11434/api/generate"

SYSTEM_PROMPT = """Extract EXACTLY these fields from shipping documents as valid JSON:
- material_type: (Steel/Cement/Aluminum/Fertilizer/Chemicals)
- tonnage: (numeric only)
- country_of_origin: (country name)
- direct_emissions: (numeric, tCO2/t)

Return ONLY JSON, no markdown. Example: {"material_type":"Steel","tonnage":100,"country_of_origin":"China","direct_emissions":5.2}"""

def extract_from_text(text):
    payload = {
        "model": "llama2",
        "prompt": f"{SYSTEM_PROMPT}\n\nText: {text[:500]}",
        "stream": False,
        "temperature": 0.1
    }
    
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        text_response = result.get('response', '')
        
        json_match = re.search(r'\{[^}]+\}', text_response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        else:
            return {'error': 'No JSON found', 'audit_flag': 'FLAG_PARSING_FAILED'}
    
    except Exception as e:
        return {'error': str(e), 'audit_flag': 'FLAG_OLLAMA_ERROR'}