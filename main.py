import os
import requests
import telebot

from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser , JsonOutputParser
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings


from dotenv import load_dotenv
from gtts import gTTS
from pydub import AudioSegment
import speech_recognition as sr
from langchain.memory import ConversationBufferWindowMemory

from vectordbsearch_tools import VectorSearchTools_chroma
import re
import json
from langchain.schema import Document  # Ensure you have this import

from web_scrape import create_vectordb
from web_scrape import get_links_and_text
from web_search import get_links_and_text_new

from langchain_chroma import Chroma
import asyncio

load_dotenv()


#app = Celery('chatbot', broker=os.getenv('CELERY_BROKER_URL'))
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)


OPENAI_API_KEY =  os.getenv('OPENAI_API_KEY') 


#embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4o-mini" ,temperature=0.1)

conversations = {}
url_vendasta ="https://www.vendasta.com/business-app-pro/"


async def add_docs(vectordb, docs):
    await vectordb.aadd_documents(documents=docs)



@staticmethod
def create_vectordb(url):
    # Assuming get_links_and_text returns a list of dictionaries with 'content' keys
    text_dicts = get_links_and_text(url)
    print(text_dicts)

    # Convert the text into Document objects
    documents = [Document(page_content=text_dict['content']) for text_dict in text_dicts]
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False
    )
    docs = splitter.split_documents(documents)
    
    persist_directory = 'chroma_db'
    
    vectordb = Chroma(embedding_function= embeddings)#, persist_directory="./chroma_db")
   
    #vectordb = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=persist_directory)
    asyncio.run(add_docs(vectordb, docs))
    
    
    return vectordb


vectordb= create_vectordb(url_vendasta)

def stand_alone(prompt, memory):
    prompt_chat = PromptTemplate(
        template="""system 
        you are a helpful assistant, provide a standalone question based on conversation history and the new question.
        Do not deviate from the core meaning of the new question.
        Do not write any explanation.
        Only provide the stand-alone question with no preamble or explanation.
        
        if user asks more information about a website from chat history, always attach the website url in your response.
        
        
        chat_historry: {memory}
        
        Question: {question} \n\n 
        
        STAND ALONE QUESTION::
        """,
        input_variables=["question", "memory"]
    )
    chain_simple = prompt_chat | llm | StrOutputParser()
    response = chain_simple.invoke({ "question": prompt, "memory": memory})
    
    return response



scraped_data = {}


def detect_and_scrape_url(message):
    # Regular expression to detect URLs
    url_pattern = re.compile(r'(https?://[^\s]+)')
    
    # Search for URLs in the message
    match = url_pattern.search(message)

    # Check if a URL was found
    if match:
        url = match.group(0)
        
        # Check if the URL has already been scraped
        if url in scraped_data:
            print(f"URL already scraped. Retrieving stored content for {url}")
            website_text = scraped_data[url]
        else:
            print(f"\nScraping {url}")
            website_text = get_links_and_text_new(url)
            # Store the scraped content
            scraped_data[url] = website_text
        
        result = {"URL": url, "text": website_text}
    else:
        result = {}

    # Convert to JSON format
    result_json = json.dumps(result)
    print(result_json)
    return result_json


def format_conversation(conversation):
    formatted_output = ""
    for message in conversation:
        role = message['role'].capitalize()
        content = message['content'].strip()
        formatted_output += f"{role}: {content}\n\n"
    return formatted_output.strip()


def get_response(prompt, stand_alone_question, data, memory_org):
    print(memory_org)
    memory= format_conversation(memory_org)
    
    print("\n\nConversation\n\n")
    print(memory) 
    # Assuming you want the last 3 messages from the list
    recent_messages = memory_org[-5:]  # Gets the last three items
    recent_messages_user = [msg['content'] for msg in recent_messages if msg['role'] == 'user']
    trimed_memory_cleaned = [text.replace('\n\n', ' ') for text in recent_messages_user]
    
    print("\n\ntrimed memory\n")
    print(trimed_memory_cleaned)

    search_keywords= f"{stand_alone_question} {prompt} {trimed_memory_cleaned}"
    new_website=detect_and_scrape_url(search_keywords)
    
    prompt_chat = PromptTemplate(
        template="""
        You are a helpful assistant on this website "https://www.vendasta.com/business-app-pro/"
        you provide information about this website.
        talk humbly. Answer the question from the provided context only. 
        Use the following pieces of context to answer the question at the end.
        your response must be according to thee chat history.
        
        Never respond in markdown format.
        
        if asked, you can also write cold outreach emails when asked. write in an effective and friendly manner without ending signatures. 
        
        this is the context :
        
        {new_website}\n\n
        
        
        {data}\n\n
        
        chat_historry: {memory}
        
        Question: {question} \n\n 
        Answer:
        """,
        input_variables=["question", "new_website", "data", "memory"]
    )
    chain_simple = prompt_chat | llm | StrOutputParser()
    response = chain_simple.invoke({ "new_website": new_website, "question": prompt, "data": data, "memory": memory})
    
    return response



def generate_response_chat(message_list, memory):
    #if VectorSearchTools_chroma:
    last_message = message_list[-1]
    try:
        stand_alone_question = stand_alone(message_list, memory)
        print("\n\nstand alone question\n")
        print(stand_alone_question)
        docs = vectordb.similarity_search(stand_alone_question)#VectorSearchTools_chroma.dbsearch(stand_alone_question)
        updated_content = last_message["content"] + "\n\n"
        
        
        print("\n\nData\n\n")
        print(docs)
    except Exception as e:
        print(f"Error while fetching: {e}")
        updated_content = last_message["content"]

    updated_message = {"role": "user", "content": updated_content}
    message_list[-1] = updated_message
    
    
    response = get_response(last_message["content"], stand_alone_question, docs, message_list)
    
    print("\n\nResponse\n\n")
    print(response)
    
    return response

def conversation_tracking(text_message, user_id):
    user_conversations = conversations.get(user_id, {'conversations': [], 'responses': []})
    user_messages = user_conversations['conversations'][-6:] + [text_message]
    user_responses = user_conversations['responses'][-6:]
    conversations[user_id] = {'conversations': user_messages, 'responses': user_responses}
    
    conversation_history = []
    for i in range(min(len(user_messages), len(user_responses))):
        conversation_history.append({"role": "user", "content": user_messages[i]})
        conversation_history.append({"role": "assistant", "content": user_responses[i]})
    
    conversation_history.append({"role": "user", "content": text_message})
    memory = ConversationBufferWindowMemory(memory_key="chat_history", input_key="question", human_prefix="User", ai_prefix="Assistant", k=6)
    response = generate_response_chat(conversation_history, memory)
    user_responses.append(response)
    conversations[user_id] = {'conversations': user_messages, 'responses': user_responses}
    return response

@bot.message_handler(content_types=["voice"])
def handle_voice(message):
    user_id = message.chat.id
    try:
        # Download the voice message file from Telegram servers
        file_info = bot.get_file(message.voice.file_id)
        file = requests.get("https://api.telegram.org/file/bot{0}/{1}".format(
            TELEGRAM_BOT_TOKEN, file_info.file_path))

        # Save the file to disk
        with open("voice_message.ogg", "wb") as f:
            f.write(file.content)

        # Use pydub to read in the audio file and convert it to WAV format
        sound = AudioSegment.from_file("voice_message.ogg", format="ogg")
        sound.export("voice_message.wav", format="wav")

        # Use SpeechRecognition to transcribe the voice message
        r = sr.Recognizer()
        with sr.AudioFile("voice_message.wav") as source:
            audio_data = r.record(source)
            text = r.recognize_google(audio_data)

        # Generate response
        replay_text = conversation_tracking(text, user_id)

        # Send the question text back to the user
        new_replay_text =  text + "\n\n" + replay_text #"You: " + text + "\n\n" + "Bot: " + replay_text
        bot.reply_to(message, new_replay_text)

        # Use Google Text-to-Speech to convert the text to speech
        tts = gTTS(replay_text)
        tts.save("voice_message.mp3")

        # Use pydub to convert the MP3 file to the OGG format
        sound = AudioSegment.from_mp3("voice_message.mp3")
        sound.export("voice_message_replay.ogg", format="mp3")

        # Send the transcribed text back to the user as a voice
        voice = open("voice_message_replay.ogg", "rb")
        bot.send_voice(message.chat.id, voice)
        voice.close()

        # Delete the temporary files
        os.remove("voice_message.ogg")
        os.remove("voice_message.wav")
        os.remove("voice_message.mp3")
        os.remove("voice_message_replay.ogg")
    except Exception as e:
        bot.reply_to(message, f"An error occurred: {str(e)}")
        print(f"Error: {str(e)}")


@bot.message_handler(func=lambda message: True)
def echo_message(message):
    user_id = message.chat.id
    if message.text == '/clear':
        conversations[user_id] = {'conversations': [], 'responses': []}
        bot.reply_to(message, "Conversations and responses cleared!")
        return
    
    response = conversation_tracking(message.text, user_id)
    bot.reply_to(message, response)

if __name__ == "__main__":
    print("Starting bot...")
    print("Bot is Active")
    bot.polling(non_stop=True)
