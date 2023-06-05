import panel as pn

def collect_messages(event):
    user_input = inp.value
    inp.value = ''
    
    # Process user input and generate assistant response
    response = process_user_message(user_input, context)
    
    # Update conversation history
    context.append({'role': 'user', 'content': user_input})
    context.append({'role': 'assistant', 'content': response})
    
    # Update display
    update_display()

def update_display():
    conversation = pn.Column()
    for item in context:
        role = item['role']
        content = item['content']
        if role == 'user':
            conversation.append(pn.Row('User:', pn.pane.Markdown(content, width=600)))
        elif role == 'assistant':
            conversation.append(pn.Row('Assistant:', pn.pane.Markdown(content, width=600, style={'background-color': '#F6F6F6'})))
    
    # Update the conversation panel
    conversation_panel.object = conversation

context = [{'role': 'system', 'content': "You are @NameofCreator"}]
inp = pn.widgets.TextInput(placeholder='Enter text here…')
button_conversation = pn.widgets.Button(name="@NameofCreator")

button_conversation.on_click(collect_messages)

conversation_panel = pn.panel('', sizing_mode='stretch_width', height=300)
dashboard = pn.Column(
    inp,
    button_conversation,
    conversation_panel
)

update_display()
dashboard.show()
