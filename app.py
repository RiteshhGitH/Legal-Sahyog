import PyPDF2
import re
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
from langchain.agents import initialize_agent, AgentType,Tool
from langchain.schema import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
import gradio as gr
import os
import pytesseract
from PIL import Image
import pickle
from langchain.agents import AgentType, initialize_agent
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import HuggingFaceHub
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.tools import Tool
# Load the CSV data as a DataFrame
df = pd.read_csv("hf://datasets/kshitij230/Indian-Law/Indian-Law.csv")
df.dropna(inplace=True)
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
index = faiss.read_index('IPC_index.faiss')
index2 = faiss.read_index('CrpC_index.faiss')
with open('IPC_N.pkl', 'rb') as f:
    flattened_data = pickle.load(f)
with open('IPC_F.pkl', 'rb') as f:
    pdf_filenames = pickle.load(f)
with open('IPC_C.pkl', 'rb') as f:
    chunk_indices = pickle.load(f)
with open('CrPC_N.pkl', 'rb') as f:
    flattened_data2 = pickle.load(f)
with open('CrPC_F.pkl', 'rb') as f:
    pdf_filenames2 = pickle.load(f)
with open('CrPC_C.pkl', 'rb') as f:
    chunk_indices2 = pickle.load(f)
# Step 3: Retrieval with Citations using PDF filename
def retrieve_faq(query):
    relevant_rows = df[df['Instruction'].str.contains(query, case=False)]
    if not relevant_rows.empty:
        response = relevant_rows.iloc[0]['Response']
        return response
    else:
        return "Sorry, I couldn't find relevant FAQs for your query."
def retrieve_info_with_citation(query, top_k=5):
    query_embedding = model.encode([query])
    D, I = index.search(query_embedding, k=top_k)

    results = []
    for i in range(min(top_k, len(I[0]))):
        if D[0][i] < 1.0:  # Relevance threshold
            chunk_index = I[0][i]
            pdf_filename = pdf_filenames[chunk_index]
            chunk_number = chunk_indices[chunk_index] + 1 
            match = flattened_data[chunk_index]
            citation = f"Source: {pdf_filename}, Chunk: {chunk_number}"
            results.append((match, citation))
        else:
            break

    if results:
        return results
    else:
        return [("I'm sorry, I couldn't find relevant information.", "Source: N/A")]


def retrieve_info_with_citation2(query, top_k=5):
    query_embedding = model.encode([query])
    D, I = index2.search(query_embedding, k=top_k)

    results = []
    for i in range(min(top_k, len(I[0]))):
        if D[0][i] < 1.0:  # Relevance threshold
            chunk_index = I[0][i]
            pdf_filename = pdf_filenames2[chunk_index]
            chunk_number = chunk_indices2[chunk_index] + 1 
            match = flattened_data2[chunk_index]
            citation = f"Source: {pdf_filename}, Chunk: {chunk_number}"
            results.append((match, citation))
        else:
            break

    if results:
        return results
    else:
        return [("I'm sorry, I couldn't find relevant information.", "Source: N/A")]

def retrieve_info(query):
    results = retrieve_info_with_citation(query)
    formatted_results = "\n\n".join([f"{i+1}. {match}\n{citation}" for i, (match, citation) in enumerate(results)])
    return formatted_results

def retrieve_info2(query):
    results = retrieve_info_with_citation2(query)
    formatted_results = "\n\n".join([f"{i+1}. {match}\n{citation}" for i, (match, citation) in enumerate(results)])
    return formatted_results
def doj_info(q):
    return """
    MINISTRY OF LAW AND JUSTICE
DEPARTMENT OF JUSTICE
----
Ministry of Law and Justice is the oldest limb of the Government of India. The Ministry functions through three integral departments -
Department of Legal Affairs, Legislative Department and the Department of Justice.
VISION OF THE DEPARTMENT OF JUSTICE
Facilitating administration of Justice that ensures easy access and
timely delivery of Justice to all
FUNCTIONS OF THE DEPARTMENT OF JUSTICE
Department of Justice performs the Administrative functions in
respect of setting up of higher courts, appointment of Judges in
higher Judiciary, maintenance and revision of the conditions and
rules of service of the Judges and issues relating to legal reforms.
The Department of Justice is also responsible jointly with the
judiciary for reduction of pendency of cases in courts. It provides
funding assistance to State Governments for modernization of
infrastructure and for projects such as computerization of
subordinate courts. Detailed functions of Department of Justice are
at Annexe.
SCHEMES UNDER THE DEPARTMENT OF JUSTICE:
2
Apart from above functions, the Department of Justice administers
various schemes to improve justice delivery.
1. Centrally sponsored scheme for development of
infrastructure for the judiciary
A Centrally Sponsored Scheme for the development of infrastructure
facilities for the judiciary is being implemented by the Department of
Justice. The scheme provides funding for construction of courtbuildings and residential accommodation for judicial officers/judges
covering both the High Courts and districts/subordinate Courts. One
of the main conditions of the scheme is that the State Government
must provide 25%of the amount against 75% released by the
Centre.
2. Gram Nayayalayas (People’s Court)
The Gram Nyayalayas Act 2008 has been enacted to provide for the
establishment of Gram Nyayalayas at the grass-root level for the
purpose of providing access to justice to the citizens at their door
steps and to ensure that opportunities for securing justice are not
denied to any citizen by reason of social, economic or other
disabilities.
The Central Government has committed to fund the initial cost in
terms of the non- recurring expenses for setting up these courts.
3. e-Courts
The Government is implementing an e-Courts Mission Mode Project
for computerization of District & Subordinate Courts in the country
and for up gradation of ICT infrastructure of the Supreme Court and
the High Courts. By virtue of this, case filing, allocation, registration,
case workflow, orders and judgements will all be ICT enabled in the
3
long run. The project was built a national judicial data grid which
enables lawyers and litigants to access case information and the
judiciary to improve case and court management.
4. National Mission for Justice Delivery and Legal Reform
The National Mission for Justice Delivery and Legal Reforms was set
up in August, 2011 to achieve the twin goals of (i) increasing access
by reducing delays and arrears; and (ii) enhancing accountability
through structural changes and by setting performance standards
and capacities.
5. Legal Aid to Poor
Assistance is provided to poor people throughout the country for
enabling them to access free legal services. The activities and free
legal services are provided through National Legal Services
Authority (NALSA) established vide National Legal Services
Authority Act, 1987.
6. Access to Justice for the marginalized
The Department of Justice is implementing two projects on ‘Access
to Justice for Marginalised People’ one of them with UNDP support.
The focus of the projects has been on empowering the poor and
marginalized, make them aware of their rights to demand legal
services, while at the same time supporting national and local justice
delivery institutions to bring justice to the poor.
7. Fast Track Courts
The Eleventh Finance Commission had recommended a scheme for
creation of Fast Track Courts (FTCs) in the country for disposal of
long pending Sessions and other cases. Fast Track Courts are set
4
up by the State Governments in consultation with the respective
High Court. Central Government provided financial assistance to
states for Fast Track Courts for eleven years from 2000-2001 to
2010-2011. In its judgment in Brij Mohan Lal vs Union of India &
Others on 19.04.2012, the Supreme Court has directed the States
that they shall not take a decision to continue the Fast Track Courts
scheme on an adhoc and temporary basis. They (States) will need to
decide either to bring the Fast Track Courts scheme to an end or to
continue the same as a permanent feature in the State. A number of
States are now continuing Fast Track Courts from their own
resources.
In the Conference of Chief Ministers and Chief Justices held in New
Delhi on 7th April, 2013, it has been resolved that the State
Governments shall, in consultation with the Chief Justices of the
respective High Courts, take necessary steps to establish suitable
number of FTCs relating to offences against women, children,
differently abled persons, senior citizens and marginalized sections
of the society, and provide adequate funds for the purpose of
creating and continuing them. Government has requested the State
Governments and the Chief Justices of the High Courts to implement
this decision.
The 14th Finance Commission has endorsed the proposal to
strengthen the judicial system in States which includes, inter-alia,
establishing 1800 FTCs for a period of five years for cases of
heinous crimes; cases involving senior citizens, women, children,
disabled and litigants affected with HIV AIDS and other terminal
ailments; and civil disputes involving land acquisition and
5
property/rent disputes pending for more than five years. The 14th
Finance Commission has urged State Governments to use
additional fiscal space provided by the Commission in the tax
devolution to meet such requirements.
JUDICIAL STRUCTURE
India has a three tier judicial structure. At the lowest level are the
District and Subordinate Courts, in over 600 administrative districts.
At the next level are the High Courts in the States. By and large,
each State has a High Court. But some states have a common High
Court. (There are a total of 24 High Courts in the country). At the
apex level is the Supreme Court of India situated at New Delhi.
1. Supreme Court of India
The Supreme Court of India comprises the Chief Justice and 30
other Judges appointed by the President of India. The Judges of the
Supreme Court are appointed by the President under Article 124 (2)
of the Constitution while the Judges of the High Courts are
appointed under Article 217 (1) of the Constitution. The President is
required to hold consultation with such of the Judges of the Supreme
Court and of the High Courts in the State as he / she may deem
necessary for the purpose. However, consultation with the Chief
Justice of India is mandatory and constitutionally a must, for
appointment of Judges other than the Chief Justice in the Supreme
Court. Supreme Court Judges retire upon attaining the age of 65
years. In order to be appointed as a Judge of the Supreme Court, a
person must be a citizen of India and must have been, for at least
6
five years, a Judge of a High Court or of two or more such Courts in
succession, or an Advocate of a High Court or of two or more such
Courts in succession for at least 10 years or he must be, in the
opinion of the President, a distinguished jurist. Provisions exist for
the appointment of a Judge of a High Court as an Ad-hoc Judge of
the Supreme Court and for retired Judges of the Supreme Court or
High Courts to sit and act as Judges of that Court.
2. High Courts
The High Court is the apex court of the State’s judicial
administration. The Judges of the High Courts are appointed by the
President under Article 217 (1) of the Constitution. There are 24
High Courts in the country, three having jurisdiction over more than
one State. Among the Union Territories Delhi alone has a High Court
of its own. Other six Union Territories come under the jurisdiction of
different State High Courts. Each High Court comprises a Chief
Justice and such other Judges as the President may, from time to
time, appoint. The Chief Justice of a High Court is appointed by the
President in consultation with the Chief Justice of India and the
Governor of the State. The procedure for appointing puisne Judges
is the same except that the Chief Justice of the High Court
concerned is also consulted. They hold office until the age of 62
years and are removable in the same manner as a Judge of the
Supreme Court. To be eligible for appointment as a Judge one must
be a citizen of India and have held a judicial office in India for ten
years or must have practised as an Advocate of a High Court or two
or more such Courts in succession for a similar period.
The transfer of Judges from one High Court to another High Court is
made by the President after consultation with the Chief Justice of
India under Article 222 (1) of the Constitution.
7
3. Subordinate Courts
Different State laws provide for different kinds of jurisdiction of
courts. Each State is divided into judicial districts presided over by a
District and Sessions Judge, which is the principal civil court of
original jurisdiction and can try all offences including those
punishable with death. The Sessions Judge is the highest judicial
authority in a district. Below him, there are Courts of civil jurisdiction,
known in different States as Munsifs, Sub-Judges, Civil Judges and
the like. Similarly, the criminal judiciary comprises the Chief Judicial
Magistrates and Judicial Magistrates of First and Second Class.
In exercise of powers conferred under proviso to Article 309 read
with Articles 233 and 234 of the Constitution, the State Government
frames rules and regulations in consultation with the High Court for
appointments, posting and promotion of District Judges. As per
Article 235, the control over subordinate courts in a State vests in
the High Court. The members of the State Judicial Service are
governed by these rules and regulations. Therefore, the service
conditions, including appointment, promotion, and reservations etc.
of judicial officers of the District/Subordinate Courts are governed by
the respective State Governments.
NATIONAL JUDICIAL APPOINTMENTS COMMISSION
The Government of India has decided to set up a National judicial
Appointments Commission (NJAC) for appointment of Judges of
Supreme Court and High Courts. The NJAC would replace the
present Collegium system of the Supreme Court for recommending
appointment of Judges in higher judiciary.
8
The Constitution Amendment Act, 2014 published on 31st December,
2014 provides for the composition and the functions of the National
Judicial Appointments Commission (NJAC). The NJAC would be
chaired by the Chief Justice of India. Its membership would include
two senior most Judges of the Supreme Court, the Union Minister of
Law & Justice, two eminent persons to be nominated by a committee
of the Prime Minister of India, the Chief Justice of India, and the
Leader of the Opposition in the House of the People, or if there is no
Leader of the Opposition, then the Leader of the single largest Opposition Party in the House of the People. Secretary (Justice) will
be the Convenor of the Commission.
GRIEVANCES AGAINST JUDICIARY
Department of Justice receives online/off line grievances from public
against judgments of the Courts, delay in their cases and against
Judges/Judicial Officers. These grievances are forwarded to the
Secretary General of Supreme Court/Registrar Generals of the
concerned High Courts for disposal at their end.
PREVIOUS VISIT OF CHINESE DELEGATION
A Chinese Delegation lead by Shri Zhang Sujun, Hon’ble Vice
Minister for Justice of People’s Republic of China, visited
Department of Justice on 26th Nov, 2012 and held discussions with
Secretary (Justice) and other senior officers of Department of
Justice. A copy of the Record of Discussion of this meeting is at
Annexe-I.
MOU BETWEEN INDIA & CHINA
A Memorandum of Understanding between the Ministry of law &
Justice of the Government of Republic of India and the Supreme
9
Peoples’ Prosecution Service of the People’s Republic of China
relating to promotion of cooperation in Legal/Judicial matters was
signed on 23rd June, 2003.
--------
""".replace('/n',' ')

ipc_tool = Tool(
    name="IPC Information Retrieval",
    func=retrieve_info,
    description="Retrieve information from the Indian Penal Code Related to query keyword(s)."
)

crpc_tool=Tool(
    name="CrPC Information Retrieval",
    func=retrieve_info2,
    description="Retrieve information from the Code of Criminal Procedure(CrPC) Related to query keyword(s)."
)

doj_tool=Tool(
    name="Department of Justice Info",
    func=doj_info,
    description="Provides Summarized Information about Department of Justice."
)
faq_tool=Tool(
    name="Commonly Asked Questions",
    func=retrieve_faq,
    description="Provides Answers to commonly asked questions related to query keyword(s)"
)
# Language model setup using Google Gemini
llm = ChatOpenAI(

    model="gpt-4o",
    temperature=1,
    max_tokens=None,
    timeout=None,
    max_retries=5
)
template="""
   You are a highly specialized legal assistant with deep knowledge of the Indian Penal Code (IPC). 
Your primary task is to retrieve and summarize legal information accurately from the IPC.pdf document provided to you. 
Your responses should be highly specific, fact-based, and free from any speculation or hallucinations. 
Always cite the exact section from the IPC when providing an answer. 
If the information is not available in the document, clearly state that and do not make any assumptions.

History: {}

User: {}

Response:
"""

agent_tools = [ipc_tool, crpc_tool, doj_tool, faq_tool]

agent = initialize_agent(
    tools=agent_tools,
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    return_intermediate_steps=True,
    handle_parsing_errors=True
)


def encode_image_to_base64(image_path):
    return pytesseract.image_to_string(Image.open(image_path))
def parse(history):
    p='\n'
    if history==[]:
        return "No Chat till now"
    for i in history:
        try:
            p+=i['role']+': '+i['content']+'\n'
        except:
            print(i)
    return p
def chatbot_response(history,query):
    l=parse(history)
    
    if query.get('files'):
        # Encode image to base64
        image_data=""
        for x in range(len(query["files"])):
            image_data += f"{x}. "+encode_image_to_base64(query["files"][x]) +"\n"
        
        # Create a multimodal message with both text and image data
        message = HumanMessage(
            content=[
                {"type": "text", "text": template.format(l,query['text'] +" System :Image(s) was added to this prompt by this user. Text Extracted from this image (Some words may be misspelled ,Use your understanding ):"+image_data)},  # Add text input
               
            ]
        )
        #k+=" System :Image(s) was added to this prompt by this user. Text Extracted from this image (Some words may be misspelled ,Use your understanding ):"+image_data
    else:
        # If no image, only pass the text
        message = HumanMessage(content=[{"type": "text", "text": template.format(l,query)}])
    # Invoke the model with the multimodal message
    result = agent.invoke([message],handle_parsing_errors=True)
    response = result['output']
    intermediate_steps = result.get('intermediate_steps', [])
    
    thought_process = ""
    for action, observation in intermediate_steps:
        thought_process += f"Thought: {action.log}\n"
        thought_process += f"Action: {str(action.tool).replace('`','')}\n"
        thought_process += f"Observation: {observation}\n\n"
    return response, thought_process.strip()
# Step 5: Gradio Interface
from gradio import ChatMessage
def chatbot_interface(messages,prompt):
    response, thought_process = chatbot_response(messages,prompt)
    #messages.append(ChatMessage(role="user", content=prompt))
    for x in prompt["files"]:
            messages.append(ChatMessage(role="user", content={"path": x, "mime_type": "image/png"}))
    if prompt["text"] is not None:
            messages.append(ChatMessage(role="user", content=prompt['text']))
    if thought_process:
        messages.append(ChatMessage(role="assistant", content=thought_process,metadata={"title": "🧠 Thought Process"}))
    messages.append(ChatMessage(role="assistant", content=response))
   
    return messages,  gr.MultimodalTextbox(value=None, interactive=True)


def vote(data: gr.LikeData):
    if data.liked:
        print("You upvoted this response: " + data.value)
    else:
        print("You downvoted this response: " + data.value)

with gr.Blocks(theme=gr.themes.Soft(),css="footer {visibility: hidden}") as iface:
   
            chatbot = gr.Chatbot(type="messages",avatar_images=("user.jpeg", "logo.jpeg"), bubble_full_width=True)  # Chatbot component to display conversation history
            query_input = gr.MultimodalTextbox(interactive=True,
                                      placeholder="Enter message or upload file...", show_label=False)
        
            
            query_input.submit(chatbot_interface, [chatbot, query_input], [chatbot,query_input])

            chatbot.like(vote, None, None)  # Adding like/dislike functionality to the chatbot

        
iface.launch(
    show_error=True
)