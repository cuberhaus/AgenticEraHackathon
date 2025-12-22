import asyncio
import json
import logging
import psycopg2
from psycopg2.extras import RealDictCursor
from langchain.prompts import ChatPromptTemplate
from langchain_google_vertexai import ChatVertexAI
from vertexai import agent_engines
from google.adk import Agent
from google.cloud import storage

# --- Configuración de logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# --- Step 1. Obtener atributos desde CloudSQL ---
def get_attributes_from_db():
    logger.info("Conectando a CloudSQL...")
    try:
        conn = psycopg2.connect(
            host="/cloudsql/qwiklabs-gcp-02-adac5bb9cc69:europe-west1:gestion-expedientes",
            dbname="postgres",
            user="postgres",
            password="MiPasswordSeguro123",
        )
        query = """
        SELECT e.*, d.*
        FROM public.estructura AS e
        JOIN public.documentacion AS d ON e.id_tipo_doc = d.id_tipo_doc
        WHERE d.id_tramite = 1;
        """
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query)
            rows = cur.fetchall()
        logger.info(f"Obtenidos {len(rows)} atributos de CloudSQL")
        logger.debug(f"Atributos: {rows}")
    except Exception as e:
        logger.error(f"Error conectando a CloudSQL: {e}")
        rows = []
    finally:
        conn.close()
    return rows

# --- Step 2. Construir especificación JSON dinámicamente ---
def build_format_instructions(rows: list[dict]) -> str:
    fields = []
    for r in rows:
        field_name = r.get("nombre_campo") or r.get("campo")
        description = r.get("descripcion", "Campo extraído del documento")
        fields.append(f'"{field_name}": "<{description}>"')
    fields_str = ",\n  ".join(fields)
    instructions = f"Por favor, devuelve un JSON con los siguientes campos:\n{{\n  {fields_str}\n}}"
    logger.debug(f"Format instructions construidas:\n{instructions}")
    return instructions

# --- Step 3. Leer TXT del bucket ---
def read_preprocessed_txt(bucket_name: str, prefix: str = "preprocessed/") -> str:
    logger.info(f"Leyendo archivos de bucket {bucket_name} con prefijo {prefix}")
    client = storage.Client()
    aggregated_text = []
    try:
        for blob in client.list_blobs(bucket_name, prefix=prefix):
            if blob.name.endswith(".txt"):
                logger.info(f"Descargando {blob.name}")
                aggregated_text.append(blob.download_as_text())
        logger.info(f"Se han leído {len(aggregated_text)} archivos de texto")
    except Exception as e:
        logger.error(f"Error leyendo bucket: {e}")
    return "\n\n".join(aggregated_text)

# --- Step 4. Tool dinámico ---
class ParseApplicationTool:
    def __init__(self):
        self.llm = ChatVertexAI(model_name="gemini-2.5-flash", temperature=0.0)

    async def run(self, document_text: str, attributes: list[dict]) -> dict:
        logger.info("Construyendo prompt para el LLM...")
        format_instructions = build_format_instructions(attributes)

        prompt = ChatPromptTemplate.from_template("""
Extrae del siguiente texto los campos indicados. 

{format_instructions}

Texto:
{document_text}
""")

        formatted_prompt = prompt.format(
            document_text=document_text,
            format_instructions=format_instructions,
        )

        logger.info("Invocando LLM...")
        try:
            response = await self.llm.ainvoke(formatted_prompt)
            logger.info("Respuesta del LLM recibida")
            logger.debug(f"Respuesta del LLM: {response}")
            return json.loads(response.content)
        except Exception as e:
            logger.error(f"Error parseando respuesta del LLM: {e}")
            return {"raw_output": getattr(response, "content", str(e))}

# --- Step 5. Root Agent ---
parse_tool = ParseApplicationTool()

async def parse_from_bucket(bucket_name: str, prefix: str = "preprocessed/") -> dict:
    logger.info("Obteniendo atributos desde la base de datos...")
    attributes = get_attributes_from_db()
    logger.info("Leyendo documentos desde bucket...")
    document_text = read_preprocessed_txt(bucket_name, prefix)
    logger.info("Ejecutando herramienta de parseo...")
    return await parse_tool.run(document_text, attributes)

agent = Agent(
    model="gemini-2.5-flash",
    name="gsp_aid_agent",
    tools=[parse_from_bucket],
)

app = agent_engines.AdkApp(agent=agent)

# --- Step 6. Runner ---
async def main():
    bucket_name = "qwiklabs-gcp-02-adac5bb9cc69-docs-intake"
    logger.info(f"Iniciando parseo de documentos en {bucket_name}/preprocessed/")
    async for event in app.async_stream_query(
        user_id="1",
        message=f"Parse documents in {bucket_name}/preprocessed/",
    ):
        logger.info(f"Evento recibido: {event}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logger.error(f"Error ejecutando main: {e}")
