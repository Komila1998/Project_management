from inference import chatbot_response

if __name__ == "__main__":
    print("Chatbot ready. Type 'quit' to exit.")
    while True:
        text = input("\nYou: ")
        if text.lower() in ["quit", "exit"]:
            break
        output = chatbot_response(text)
        print("Bot:", output["response"])
