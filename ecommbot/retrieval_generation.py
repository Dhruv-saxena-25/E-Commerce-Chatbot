from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationChain
from ecommbot.data_ingestion import data_ingestion


def generation(vstore):

    retriever = vstore.as_retriever(search_kwargs={"k": 3})

    PRODUCT_BOT_TEMPLATE = """ Your ecommercebot bot is an expert in product recommendations and customer queries.
    It analyzes product titles and reviews to provide accurate and helpful responses.
    Ensure your answers are relevant to the product context and if entered infromation is off-toppic
    please ensure that the information is out of the. Your responses should be concise and informative.

    CONTEXT: {context}

    QUESTION: {question}

    YOUR ANSWER:
    """

    prompt = ChatPromptTemplate.from_template(PRODUCT_BOT_TEMPLATE)

    memory = ConversationBufferWindowMemory(k=2)
    
    llm = ChatOpenAI(temperature= 0.6)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

if __name__=='__main__':
    vstore = data_ingestion("done")
    chain  = generation(vstore)
    print(chain.invoke("can you tell me the best bluetooth buds?"))


