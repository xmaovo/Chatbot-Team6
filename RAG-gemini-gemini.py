import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
import tempfile
import time
import google.generativeai as genai


persona = '''
You are a professional AI assistant specialized in financial analysis of SEC 10-K filings from public companies. Your primary role is to help users understand the contents of annual reports (10-K) filed by Alphabet, Amazon, and Microsoft.

Your responses must follow these principles:
- Only answer questions using information from the provided documents.
- If a question is outside the scope of these documents and lacks sufficient context, respond first with:
  "I don't have enough information to answer this question based on the documents provided."
  Then, based on your understanding of the user's question, politely ask a clarification question that is specific to the topic or term the user mentioned.
  For example, if the user asks about "ESG trends", you may ask: "Could you specify which ESG component (Environmental, Social, or Governance) and time period you're referring to?"
  If the user asks about acquisitions, you may ask: "Are you referring to a specific business area, such as AI or logistics?"

- Do not use external knowledge or assumptions. Avoid hallucinations at all costs.
- Prefer factual accuracy over fluency. A concise, document-based response is better than a speculative one.
- When possible, cite specific document sections (e.g., 'Alphabet 10-K, Part I, Item 1: Business').

Additional instructions:
- Support users in comparing companies, analyzing risks, understanding ESG strategies, executive compensation, and segment-level performance—based strictly on the documents.
- Treat forward-looking statements cautiously; clarify if the source is a projection or estimate.
- If user input is vague or overly broad, always follow up with a clarification request that reflects the context of the question.

Maintain a professional, neutral tone and clearly communicate any limitations in the available data.
'''


template = """
{persona}
        
Chat History:
<history>

User: What is the primary business segment driving revenue for Amazon, Google, and Microsoft, and how do they differ?
Assistant: Amazon primarily earns revenue through its North America and International retail segments, but Amazon Web Services (AWS) is its most profitable division. Google (Alphabet) derives most of its revenue from advertising across Google Search, YouTube, and Google Network, with Google Cloud as a growing segment. Microsoft generates its highest revenue from its cloud computing services, including Azure, followed by Office software and Windows products.

User: How are each of the three companies investing in artificial intelligence, and what unique AI products or services do they highlight?
Assistant: Alphabet focuses heavily on AI with its Gemini models integrated across Google services like Search, Gmail, and YouTube. Microsoft invests deeply in OpenAI partnerships, embedding Copilot into Office 365, Azure, and GitHub. Amazon offers generative AI tools through AWS Bedrock and Titan models, along with custom AI chips like Trainium and Inferentia.

User: Which geographic regions or international segments contribute most significantly to each company's growth strategy?
Assistant: Amazon's International segment supports global expansion but operates with thinner margins than North America. Microsoft reports strong international performance from enterprise cloud services and global datacenters. Alphabet highlights international user growth, especially in emerging markets, as a key driver for advertising and YouTube.

User: What are the major risks each company identifies as critical to its long-term sustainability and competitive edge?
Assistant: All three companies list regulatory pressures, cybersecurity threats, and antitrust scrutiny as top risks. Amazon also notes workforce challenges and logistics costs; Microsoft highlights intellectual property concerns; and Alphabet cites changes to advertising policies and evolving AI regulation.

User: How does each company describe its commitment to sustainability, and what specific environmental goals have they publicly committed to?
Assistant: Microsoft aims to be carbon negative, water positive, and zero waste by 2030. Alphabet targets operating on 24/7 carbon-free energy by 2030 across all locations. Amazon emphasizes The Climate Pledge, aiming for net-zero carbon by 2040 and significant investments in renewable energy and electric delivery vehicles.

User: How does each company balance innovation with ethical concerns, especially around data privacy, AI use, or labor practices?
Assistant: Alphabet follows its AI Principles to guide ethical AI development and has emphasized responsible use in Gemini deployments. Microsoft leads with AI safety research and transparent frameworks, including tools for responsible AI governance. Amazon highlights workforce safety investments, privacy controls in devices like Alexa, and ethical AI within its cloud services.

{chat_history}
</history>

Given the context information and not prior knowledge, answer the following question:
Question: {user_input}
"""


# Set Google API key (replace with your key or use an env variable)
GOOGLE_API_KEY = '' # "YOUR_GOOGLE_API_KEY"  # Replace with your actual Gemini API key
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.5)


st.set_page_config(page_title="Chat with Your PDFs (Gemini)")

st.title("📄💬 Chat with Your PDFs (Gemini)")

# File uploader
uploaded_files = st.file_uploader("Upload PDFs", accept_multiple_files=True, type=["pdf"])

if uploaded_files:
    documents = []
    
    # Process PDFs if vector store doesn't exist in session state
    if "vector_store" not in st.session_state:
        with st.spinner("Processing your PDFs..."):
            # Create a temporary directory to save the uploaded files
            with tempfile.TemporaryDirectory() as temp_dir:
                for file in uploaded_files:
                    # Save the uploaded file to a temporary file
                    temp_file_path = os.path.join(temp_dir, file.name)
                    with open(temp_file_path, "wb") as f:
                        f.write(file.getbuffer())
                    
                    # Load the PDF
                    loader = PyPDFLoader(temp_file_path)
                    documents.extend(loader.load())

                # Split documents into chunks
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                docs = text_splitter.split_documents(documents)

                # Generate embeddings and store in FAISS
                # ------------------------------------------------- #
                
                # use your embeddings model here
                st.session_state.vector_store = FAISS.from_documents(docs, embeddings)
                # ------------------------------------------------- #


        st.success("✅ PDFs uploaded and processed! You can now start chatting.")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    user_input = st.chat_input("Ask a question about your PDFs...")
    
    if user_input:
        # Immediately add user message to chat history and display it
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Display the user message immediately
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Configure retriever with more advanced parameters
        retriever = st.session_state.vector_store.as_retriever(
            search_type="mmr",  # Maximum Marginal Relevance
            search_kwargs={"k": 8, "fetch_k": 40, "lambda_mult": 0.8}  # Adjust these parameters as needed
        )
        
        # Get chat history for context
        chat_history = ""
        if len(st.session_state.messages) > 1:  # If there are previous messages
            for i, msg in enumerate(st.session_state.messages[:-1]):  # Exclude the current user message
                role = "User" if msg["role"] == "user" else "Assistant"
                chat_history += f"{role}: {msg['content']}\n\n"
        
        # Create a custom prompt template with chat history
        
        
        # Create the QA chain with the custom prompt
        # The RetrievalQA chain will automatically handle getting the context from the retriever
        # and formatting it with the prompt template
        qa_chain = RetrievalQA.from_chain_type(
            ## use your llm model here
            llm=llm,
            retriever=retriever,
            chain_type="stuff",  # "stuff" chain type puts all retrieved documents into the prompt context
            return_source_documents=True,  # Return source documents for reference
            verbose = True,
            chain_type_kwargs={
                # "prompt": CUSTOM_PROMPT,  # Use the custom prompt
                "verbose": True  # Enable verbose mode to see the full prompt
            }
        )
        # ------------------------------------------------- #
        
        # Get response from the chatbot with spinner
        with st.spinner("Thinking..."):
            # The RetrievalQA chain automatically:
            # 1. Takes the query
            # 2. Retrieves relevant documents using the retriever
            # 3. Formats those documents as the context in the prompt
            # 4. Sends the formatted prompt to the LLM
            response = qa_chain.invoke({
                "query": template.format(
                    persona=persona,
                    user_input=user_input,
                    chat_history=chat_history
                ),
            })
            
            # For debugging, you can see what's in the response
            # st.write("Response keys:", list(response.keys()))
            
            # Display retrieved chunks in an expander if source documents are available
            if "source_documents" in response:
                with st.expander("View Retrieved Chunks (Context)"):
                    for i, doc in enumerate(response["source_documents"]):
                        st.markdown(f"**Chunk {i+1}**")
                        st.markdown(f"**Content:** {doc.page_content}")
                        st.markdown(f"**Source:** Page {doc.metadata.get('page', 'unknown')}")
                        st.markdown("---")
            
            response_text = response["result"]
        # Display assistant response
        # with st.chat_message("assistant"):
        #     st.markdown(response_text)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            # Simulate streaming with an existing string
            for chunk in response_text.split():
                full_response += chunk + " "
                message_placeholder.markdown(full_response)
                time.sleep(0.05)  # Small delay to simulate streaming
                
        # Store assistant response in session state
        st.session_state.messages.append({"role": "assistant", "content": response_text})
else:
    st.info("Please upload PDF files to begin.")
