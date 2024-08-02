from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import RetrievalQA 
from ecommbot.data_ingestion import data_ingestion




def generation(vstore):
    retriever = vstore.as_retriever(search_kwargs={"k": 3})
    memory = ConversationBufferWindowMemory(k=6)

    PRODUCT_BOT_TEMPLATE = """
    Your ecommercebot bot is an expert in product recommendations and customer queries.
    It analyzes product titles and reviews to provide accurate and helpful responses.
    Ensure your answers are relevant to the product context and refrain from straying off-topic.
    Your responses should be concise and informative.

    CONTEXT:
    {context}

    QUESTION: {question}

    YOUR ANSWER:
    
    """

    # prompt = ChatPromptTemplate.from_template(PRODUCT_BOT_TEMPLATE)
    prompt = PromptTemplate(template= PRODUCT_BOT_TEMPLATE, input_variable= ["context", "question"])

    llm = ChatOpenAI()
    chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type= "stuff",
                                        retriever= retriever,
                                        input_key= "query",
                                        memory= memory,
                                        chain_type_kwargs= {"prompt": prompt})

    # chain = (
    #     {"context": retriever, "question": RunnablePassthrough()}
    #     | prompt
    #     | llm
    #     | StrOutputParser()
    #     )

    return chain


if __name__=='__main__':
    vstore = data_ingestion("done")
    chain  = generation(vstore)
    # print(chain.invoke("can you tell me the best bluetooth buds?"))
    result= chain("can you tell me the best bluetooth buds?")
    print(result['result'])

   


