import sys
from agent_with_langchain import agent as langchain_app
from agent_with_langgraph import app as langgraph_app

# Set the way to "langchain" or "langgraph"
WAY = "langgraph"

def main():
    print("Mini Agentic RAG System")
    print(f"Mode: {WAY}")
    print("Type 'exit' or 'quit' to stop.")
    print("-" * 30)

    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break

            if not user_input.strip():
                continue

            print("Agent: Processing...")


            # LANGCHAIN MODE

            if WAY == "langchain":
                inputs = {"messages": [("user", user_input)]}
                final_response = None

                
                for output in langchain_app.stream(inputs, stream_mode="values"):
                    if "messages" in output:
                        last = output["messages"][-1]
                        final_response = last.content

                print(f"\nAgent: {final_response}\n")




            # LANGGRAPH MODE
            elif WAY == "langgraph":
                inputs = {"question": user_input}
                final_response = None

                for output in langgraph_app.stream(inputs):
                    for node, state in output.items():
                        print(f"Finished Node: {node}")
                        if "generation" in state:
                            final_response = state["generation"]

                print(f"\nAgent: {final_response}\n")

            else:
                print(f"Unknown mode: {WAY}")
                break

            print("-" * 30)

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except EOFError:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            break


if __name__ == "__main__":
    main()
