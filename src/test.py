# --- LangChain Pydantic Parsing ---
# 1. Set up the Pydantic parser
parser = PydanticOutputParser(pydantic_object=ApplicationData)
 
# 2. Set up the LLM
# Using a powerful model as requested. Ensure it's available in your region.
# Note: "gemini-2.5-pro" is hypothetical; using "gemini-1.5-pro-latest".
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
 
# 3. Create the prompt
prompt_template = """
You are an expert in processing Spanish public administration documents.
Analyze the following text extracted from a certificate document and extract the relevant information.
Format your output as a JSON object that strictly follows the provided schema.
 
{format_instructions}
 
Here is the text to analyze:
---
{document_text}
---
"""
prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["document_text"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)
 
# 4. Create the chain and invoke it
chain = prompt | model | parser
parsed_result = chain.invoke({"document_text": document.text})
 
print("Structured data generated successfully.")
parsed_result.model_dump_json(indent=2)
 
 
# --- Pydantic Models for Structured Output ---
class Certificado(BaseModel):
    numero_certificado: str = Field(description="Certificate number")
    fecha_emision: str = Field(description="Date of emission in YYYY-MM-DD format")
    codigo_csv: str = Field(description="Secure Verification Code (CSV)")
    organo_solicitante: str = Field(description="Requesting body")
    fecha_solicitud: str = Field(description="Date of request in YYYY-MM-DD format")
    finalidad: str = Field(description="Purpose of the certificate")
    nif: str = Field(description="Tax identification number (NIF) of the person")
    nombre_completo: str = Field(description="Full name of the person")
 
class ApplicationData(BaseModel):
    certificado: Certificado = Field(description="Details of the certificate")
 