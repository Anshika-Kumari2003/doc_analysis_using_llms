import gradio as gr
from gtts import gTTS
import os

# Function to generate answer and audio
def answer_and_speak(input_text):
    answer = f"Your answer is: {input_text}"  # Example: reversing text as a dummy response

    # Convert answer to speech using gTTS
    tts = gTTS(answer)
    audio_path = "output.mp3"
    tts.save(audio_path)

    return answer, audio_path

# Gradio Interface
with gr.Blocks() as demo:
    with gr.Row():
        user_input = gr.Textbox(label="Enter Question")
        submit_btn = gr.Button("Submit")

    with gr.Row():
        output_text = gr.Textbox(label="Generated Answer")
        play_audio = gr.Audio(label="Click Play to Hear", type="filepath")

    submit_btn.click(fn=answer_and_speak, inputs=user_input, outputs=[output_text, play_audio])

demo.launch()
