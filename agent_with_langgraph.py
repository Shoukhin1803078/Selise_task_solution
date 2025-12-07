import os
from typing import Annotated, List, Literal, TypedDict

from langchain_chroma import Chroma
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv

load_dotenv()


EMBEDDING_MODEL = AzureOpenAIEmbeddings(
    azure_deployment=os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"],
    openai_api_version=os.environ["AZURE_OPENAI_EMBEDDING_API_VERSION"],
)

def get_llm():
    return AzureChatOpenAI(
        azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
        api_version=os.environ["AZURE_OPENAI_CHAT_API_VERSION"],
        temperature=0
    )


def get_retriever():
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=EMBEDDING_MODEL
    )
    return vectorstore.as_retriever()


class GraphState(TypedDict):
    """
    Represents the state of our graph.
    """
    question: str
    generation: str
    documents: List[str]


#  Nodes 

def retrieve(state):
    print("---RETRIEVE---")
    question = state["question"]
    retriever = get_retriever()
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}

def generate(state):
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    
    llm = get_llm()
    
    prompt = ChatPromptTemplate.from_template(
        """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
        
        Question: {question} 
        Context: {context} 
        Answer:"""
    )
    
    chain = prompt | llm | StrOutputParser()
    generation = chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}

def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question.
    """
    print("---CHECK DOCUMENT RELEVANCE---")
    question = state["question"]
    documents = state["documents"]
    
    llm = get_llm()

    class Grade(BaseModel):
        """Binary score for relevance check."""
        binary_score: str = Field(description="Relevance score 'yes' or 'no'")

    structured_llm_grader = llm.with_structured_output(Grade)

    system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
    
    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
        ]
    )
    
    retrieval_grader = grade_prompt | structured_llm_grader
    
    filtered_docs = []
    for d in documents:
        score = retrieval_grader.invoke({"question": question, "document": d.page_content})
        grade = score.binary_score
        if grade == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            continue
            
    return {"documents": filtered_docs, "question": question}

def transform_query(state):
    """
    Transform the query to produce a better question.
    """
    print("---TRANSFORM QUERY---")
    question = state["question"]
    documents = state["documents"]
    
    llm = get_llm()
    
    system = """You are a question re-writer that converts an input question to a better version that is optimized \n 
     for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""
     
    re_write_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Here is the initial question: \n\n {question} \n Formulate an improved question."),
        ]
    )
    
    question_rewriter = re_write_prompt | llm | StrOutputParser()
    better_question = question_rewriter.invoke({"question": question})
    
    return {"documents": documents, "question": better_question}


# Conditional Edges

def decide_to_generate(state):
    """
    Determines whether to generate an answer, or re-generate a question.
    """
    print("---ASSESS GRADED DOCUMENTS---")
    filtered_documents = state["documents"]
    
    if not filtered_documents:
        # All documents have been filtered check_relevance
        # Will re-generate a new query
        print("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---")
        return "transform_query"
    else:
        # RAG have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"

def grade_generation_v_documents_and_question(state):
    """
    Determines whether the generation is grounded in the document and answers question.
    """
    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    
    llm = get_llm()
    
    class GradeHallucinations(BaseModel):
        """Binary score for hallucination check."""
        binary_score: str = Field(description="Answer is grounded in the facts, 'yes' or 'no'")
        
    structured_llm_grader = llm.with_structured_output(GradeHallucinations)
    
    system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
     Give a binary score 'yes' or 'no'. 'yes' means that the answer is grounded in / supported by the set of facts."""
     
    hallucination_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
        ]
    )
    
    hallucination_grader = hallucination_prompt | structured_llm_grader
    
    score = hallucination_grader.invoke({"documents": documents, "generation": generation})
    grade = score.binary_score
    
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        print("---GRADE GENERATION vs QUESTION---")
        
        class GradeAnswer(BaseModel):
            """Binary score to check answer addresses question."""
            binary_score: str = Field(description="Answer addresses the question, 'yes' or 'no'")
            
        structured_llm_grader_answer = llm.with_structured_output(GradeAnswer)
        
        system_answer = """You are a grader assessing whether an answer addresses / resolves a question \n 
         Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""
         
        answer_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_answer),
                ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
            ]
        )
        
        answer_grader = answer_prompt | structured_llm_grader_answer
        score_answer = answer_grader.invoke({"question": question, "generation": generation})
        grade_answer = score_answer.binary_score
        
        if grade_answer == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"


# Graph 
workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("transform_query", transform_query)

workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
    },
)
workflow.add_edge("transform_query", "retrieve")
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate", 
        "useful": END,
        "not useful": "transform_query",
    },
)

app = workflow.compile()

# === VISUALIZATION CODE ===
# Display the graph structure
try:
    from IPython.display import Image, display
    display(Image(app.get_graph().draw_mermaid_png()))
except Exception:
    # If IPython is not available, save to file
    print("Saving graph visualization to 'workflow_graph.png'...")
    with open("workflow_graph.png", "wb") as f:
        f.write(app.get_graph().draw_mermaid_png())
    print("Graph saved successfully!")