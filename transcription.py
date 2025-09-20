from dotenv import load_dotenv
from openai import OpenAI
import os


# Charger la clé API
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def transcribe_audio(audio_path: str) -> str:
    with open(audio_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
    return transcript.text

if __name__ == "__main__":
    audio_file = "Route des Lanots.m4a"  # ton fichier test
    result = transcribe_audio(audio_file)

    print("Transcription :\n")
    print(result)

# --- Config ---
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # modèle conseillé pour coût/latence

SYSTEM_PROMPT = """Tu es un assistant qui synthétise des consultations médicales
à partir d'une transcription. Tu n'inventes rien : tu ne rapportes que ce qui est
présent dans le texte. Tu n'émets pas d'avis médical définitif ; tu résumes.
Sortie EXCLUSIVEMENT en JSON valide UTF-8, sans texte autour, selon ce schéma:

{
  "context": "string",
  "symptoms": ["..."],
  "onset_duration": "string",
  "exam_findings": "string",
  "assessment": "string",
  "plan": "string",
  "safety_net": "string",
  "follow_up": "string"
}

Contrainte: si une rubrique est absente de la transcription, mets une chaîne vide
ou une liste vide, sans inventer d'information.
"""


def summarize_transcript(transcript_text: str, model: str = DEFAULT_MODEL):
    """Appelle le LLM pour obtenir un JSON structuré."""
    # Utilise Chat Completions (SDK v1+)
    # Docs générales API : https://platform.openai.com/docs/ (référence) 
    completion = client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Voici la transcription brute de la consultation (français ou anglais).\n\n"
                    f"--- TRANSCRIPTION START ---\n{transcript_text}\n--- TRANSCRIPTION END ---"
                ),
            },
        ],
    )

    return completion.choices[0].message.content

resumer = summarize_transcript(result, DEFAULT_MODEL)
