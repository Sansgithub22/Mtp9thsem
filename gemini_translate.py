from google import genai

# 🔑 PUT YOUR API KEY HERE
API_KEY = "AIzaSyAsEJuorCOriPHK7pWXHc6QKgPQivEzuL0"

client = genai.Client(api_key=API_KEY)

def translate_hi_to_bho(sentence: str) -> str:
    prompt = f"""
You are a translation assistant specialized in Indian languages.

Translate the following sentence from Hindi to Bhojpuri.
Only return the translated sentence, no explanation.

Hindi: {sentence}
Bhojpuri:
"""
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )
    return response.text.strip()

if __name__ == "__main__":
    src = "मैं स्कूल जा रहा हूँ"
    bho = translate_hi_to_bho(src)
    print("Hindi   :", src)
    print("Bhojpuri:", bho)
