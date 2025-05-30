import gradio as gr
import mindspore
from mindnlp.transformers import MiniCPM3ForCausalLM, MiniCPM3Tokenizer
from mindnlp.transformers import TextIteratorStreamer
from threading import Thread
from mindspore._c_expression import disable_multi_thread


disable_multi_thread()

# Loading the tokenizer and model from Modelers's model hub.
model_id = "MindSpore-Lab/MiniCPM3-4B-FP16"
tokenizer = MiniCPM3Tokenizer.from_pretrained(model_id, mirror='modelers', ms_dtype=mindspore.float16)
model = MiniCPM3ForCausalLM.from_pretrained(model_id, mirror='modelers', ms_dtype=mindspore.float16)

system_prompt = "You are a helpful and friendly chatbot"

def build_input_from_chat_history(chat_history, msg: str):
    messages = [{'role': 'system', 'content': system_prompt}]
    for user_msg, ai_msg in chat_history:
        messages.append({'role': 'user', 'content': user_msg})
        messages.append({'role': 'assistant', 'content': ai_msg})
    messages.append({'role': 'user', 'content': msg})
    return messages

# Function to generate model predictions.
def predict(message, history):
    # Formatting the input for the model.
    messages = build_input_from_chat_history(history, message)
    input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="ms",
            tokenizer=True
        )
    streamer = TextIteratorStreamer(tokenizer, timeout=120, skip_prompt=True, skip_special_tokens=True)
    generate_kwargs = dict(
        input_ids=input_ids,
        streamer=streamer,
        max_new_tokens=1024,
        do_sample=True,
        top_p=0.7,
        temperature=0.7,
    )
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()  # Starting the generation in a separate thread.
    partial_message = ""
    for new_token in streamer:
        partial_message += new_token
        if '</s>' in partial_message:  # Breaking the loop if the stop token is generated.
            break
        yield partial_message


# Setting up the Gradio chat interface.
gr.ChatInterface(predict,
                 title="MiniCPM3",
                 description="问几个问题",
                 examples=['推荐5个北京的景点。']
                 ).launch()  # Launching the web interface.
