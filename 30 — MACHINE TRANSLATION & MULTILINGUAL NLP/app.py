from flask import Flask, render_template, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# --------------------------------------------------
# Flask application
# --------------------------------------------------

app = Flask(__name__)


# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# --------------------------------------------------
# Model
# --------------------------------------------------

MODEL_NAME = "facebook/nllb-200-distilled-600M"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME
)

model = model.to(device)
model.eval()


# --------------------------------------------------
# Supported languages
# --------------------------------------------------

LANGUAGES = {
    "English": "eng_Latn",
    "Tamil": "tam_Taml",
    "French": "fra_Latn",
    "German": "deu_Latn",
    "Japanese": "jpn_Jpan",
}


# --------------------------------------------------
# Translation function
# --------------------------------------------------

def translate_text(
    text,
    source_language,
    target_language
):

    src_code = LANGUAGES[source_language]
    tgt_code = LANGUAGES[target_language]

    tokenizer.src_lang = src_code

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        translated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(
                tgt_code
            ),
            num_beams=4,
            max_new_tokens=128
        )

    translation = tokenizer.batch_decode(
        translated_tokens,
        skip_special_tokens=True
    )[0]

    return translation


# --------------------------------------------------
# Home page
# --------------------------------------------------

@app.route("/")
def home():

    return render_template("index.html")


# --------------------------------------------------
# Translation API
# --------------------------------------------------

@app.route("/translate", methods=["POST"])
def translate():

    data = request.get_json()

    text = data.get("text", "").strip()

    source_language = data.get(
        "source_language"
    )

    target_language = data.get(
        "target_language"
    )


    # Empty input
    if not text:

        return jsonify({
            "error": "Please enter some text."
        }), 400


    # Invalid source language
    if source_language not in LANGUAGES:

        return jsonify({
            "error": "Invalid source language."
        }), 400


    # Invalid target language
    if target_language not in LANGUAGES:

        return jsonify({
            "error": "Invalid target language."
        }), 400


    # Same language
    if source_language == target_language:

        return jsonify({
            "translation": text
        })


    try:

        result = translate_text(
            text,
            source_language,
            target_language
        )

        return jsonify({
            "translation": result
        })


    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


# --------------------------------------------------
# Run application
# --------------------------------------------------

if __name__ == "__main__":

    app.run(
        debug=True
    )