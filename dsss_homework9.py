from typing import Final
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackContext
import torch
from transformers import pipeline

API_TOKEN: Final = '7815395406:AAEevGa9i66pqMwM1om2dIa3trb03KJBrcU'
BOT_USERNAME: Final = '@dsss_ex9_llm_bot'

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hello, I'm your DSSS LLM Bot! I am using TinyLlama-1.1B-Chat-v1.0 model to answer your questions.")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hello, I'm your DSSS LLM Bot! Please type something so I can for example answer your questions.")


# Responses
# def handle_response(text: str) -> str:
#     text = text.lower()
#
#     if 'hello' in text:
#         return "Hey there!"
#
#     if "how are you" in text:
#         return "I'm fine, thank you!"
#
#     return "I am dumb. :("


def llm_handle_response(text: str):
    pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16,
                    device_map="auto")

    # We use the tokenizer's chat template to format each message - see https://huggingface.co/docs/transformers/main/en/chat_templating
    messages = [
        {
            "role": "system",
            "content": "Act in the following as a funny and friendly chatbot.",
        },
        {"role": "user", "content": text},
    ]
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)

    generated_text = outputs[0]["generated_text"]
    # Bereinige die Ausgabe
    clean_text = "\n".join(generated_text.splitlines()[5:])

    return clean_text


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message_type: str = update.message.chat.type
    text: str = update.message.text  # Incoming message, which we can process

    print(f'User ({update.message.chat.id}) in {message_type}: "{text}" ')  # user-id who is sending the message and where (private chat or group) and which type of chat

    if message_type == 'group':  # in this case the bot will run in a group and not in a private chat
        if BOT_USERNAME in text:
            new_text: str = text.replace(BOT_USERNAME, '').strip()  # This will remove the bot name
            response: str = llm_handle_response(new_text)
        else:
            return  # Because the bot should not answer unless we give its username

    else:  # privater Chat soll immer antworten einfach
        response: str = llm_handle_response(text)

    print("Bot: ", response)
    await update.message.reply_text(response)


async def error(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print(f'Update {update} caused error {context.error}')



if __name__ == '__main__':
    print('Starting telegram bot...')
    app = Application.builder().token(API_TOKEN).build()

    # Commands
    app.add_handler(CommandHandler('start', start_command))
    app.add_handler(CommandHandler('help', help_command))

    # Messages
    app.add_handler(MessageHandler(filters.TEXT, handle_message))

    # Errors
    app.add_error_handler(error)

    print('Polling...')
    app.run_polling(poll_interval=3)