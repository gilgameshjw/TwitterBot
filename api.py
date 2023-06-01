
import openai
from flask import Flask, render_template, request
import time

app = Flask(__name__)

openai.api_key = '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'  # Replace with your OpenAI API key


chat_history = []

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        user_input = request.form['user_input']
        # Send user_input to ChatGPT and get the response
        response, computation_time, price = chat_with_gpt(user_input)
        # Add the input, output, computation time, and price to chat_history
        chat_history.append({
            'input': user_input,
            'output': response,
            'computation_time': computation_time,
            'price': price
        })
    return render_template('index.html', chat_history=chat_history)

def chat_with_gpt(user_input):
    start_time = time.time()
    response = openai.Completion.create(
        engine='text-davinci-003',
        prompt=user_input,
        max_tokens=50,
        temperature=0.7,
        n = 1,
        stop=None,
    )
    end_time = time.time()
    computation_time = end_time - start_time
    price = computation_time * 0.000048  # Cost per second with text-davinci-003 engine

    response_text = response.choices[0].text.strip()
    return response_text, computation_time, price

if __name__ == '__main__':
    app.run(debug=True)

