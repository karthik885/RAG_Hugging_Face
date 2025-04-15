import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

st.title("GPT-2 Text Generator")

prompt = st.text_input("Enter a prompt", "Once upon a time...")
length = st.slider("Max length", 20, 200, 50)
temperature = st.slider("Temperature", 0.5, 1.5, 0.7)

if st.button("Generate"):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs["input_ids"], max_length=length, temperature=temperature, do_sample=True)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    st.write(result)
