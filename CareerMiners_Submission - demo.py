# Databricks notebook source
# MAGIC %sql
# MAGIC -- create table workspace.default.posting_cleaned
# MAGIC -- using delta as
# MAGIC create or replace temp view posting_no_nulls as
# MAGIC
# MAGIC select job_id, company_name, title, description from workspace.default.postings
# MAGIC where job_id is not null and company_name is not null and title is not null and description is not null
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC create table workspace.default.posting_cleaned
# MAGIC using delta as
# MAGIC
# MAGIC with cte as(select job_id, count(job_id) from posting_no_nulls
# MAGIC group by 1
# MAGIC having count(job_id) > 1
# MAGIC )
# MAGIC select * from posting_no_nulls
# MAGIC where job_id not in (select job_id from cte)

# COMMAND ----------

# DBTITLE 1,checking if there are any nulls
# MAGIC %sql
# MAGIC select job_id, count(job_id) from workspace.default.posting_cleaned
# MAGIC group by 1
# MAGIC having count(job_id) > 1

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(*) from workspace.default.posting_cleaned

# COMMAND ----------

# DBTITLE 1,Formatted dataset
# MAGIC %sql
# MAGIC select * from workspace.default.posting_cleaned

# COMMAND ----------

# DBTITLE 1,Create Vector Index for the above table
from databricks import vector_search

# Define the parameters for the vector index creation
index_name = "career_miners_description_index"
source_table = "workspace.default.posting_cleaned"
text_column = "description"
embedding_model = "databricks-bge-large-en"
options = {
    "delta_sync": "true",
    "schema_evolution": "true"
}

# Create the vector index using the Databricks Foundation Embedding model
vector_search.create_vector_index(
    index_name=index_name,
    source_table=source_table,
    text_columns=[text_column],
    embedding_model=embedding_model,
    options=options
)

# COMMAND ----------

# DBTITLE 1,Add libraries
!pip install mlflow[genai]>=2.9.0
!pip install databricks-vectorsearch

# COMMAND ----------

# DBTITLE 1,Restart the Databricks Kernel After Installing the Packages
dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Import Necessary Packages
import mlflow.deployments
import json
import os
from databricks.vector_search.client import VectorSearchClient



# COMMAND ----------

# DBTITLE 1,Using dbrx model for getting the required skills for the user input
def getLLMResponse(userinput):
    client = mlflow.deployments.get_deploy_client("databricks")
    endpoints = client.list_endpoints() 
    completions_response = client.predict(
        endpoint="databricks-dbrx-instruct",
        inputs={
        "messages": [
            {
                "role": "system",
                "content": "As a job search bot assistant AI, you have to generate a list of relevant skill sets from the user prompt or user query. Extract the skills set. Please adhere to following instructions: - Read carefully each of the user's query and identify key results like skills. - If any of these details are missing in the users query, use 'NA' for that specific information in the output. - Compile a list of relevant skills. Deliver your Output as a list of JSON objects, with no deviation. To ensure clarity and consistency, refrain from including any leading indicators like 'Output:'. For Example: Comments: I am a software engineer specializing in devops looking for a job. Output: {\"skills\"  : [\"Databricks\", \"Software Engineering\", \"Devops\", \"Kubernetes\", \"GitLab\", \"Jenkins\", \"Golang\", \"Docker\"]}."
            },
            {
                "role": "user",
                "content": userinput
            }
        ],
        "max_tokens": 4000
    }
    )
    response = completions_response.choices[0]['message']['content']
    return response


# COMMAND ----------

# DBTITLE 1,Get the User Skills Based on the user Input
def getUserSkill(userinput):
    response = getLLMResponse(userinput)
    skills = json.loads(response)
    print(f'response from LLM, {skills}')
    skills_list = skills['skills']
    skill_set  =  ', '.join(skills_list)
    print(f'Skills Required : {skill_set}')
    return skill_set


# COMMAND ----------

# DBTITLE 1,initialization and results
def getJobs(userinput, noOfJobs):

    workspace_url = os.environ.get("WORKSPACE_URL")
    sp_client_id = os.environ.get("SP_CLIENT_ID")
    sp_client_secret = os.environ.get("SP_CLIENT_SECRET")

    vsc = VectorSearchClient(
        workspace_url=workspace_url,
        service_principal_client_id=sp_client_id,
        service_principal_client_secret=sp_client_secret
    )

    index = vsc.get_index(endpoint_name="hackathon_job_miners", index_name="workspace.default.career_miners_description_index")
    skill_set = getUserSkill(userinput)
    results = index.similarity_search(num_results=noOfJobs, columns=["job_id", "title", "company_name" , "description"], query_text=skill_set)
    job_results = results['result']['data_array']
    return job_results


# COMMAND ----------

# DBTITLE 1,Results from our dataset for the relevent skills
def showjobs(userinput, noOfJobs):
    job_results = getJobs(userinput, noOfJobs)
    print(f"Top {noOfJobs} matches on the user query:")
    print("\n\n----------------------------------------------")
    print("\nJOB_ID \t Title \t Company \t Job Description \t Relevancy Score")
    print("\n----------------------------------------------")
    for job in job_results:
        #column_index = job[0].index("job_id")
        print(f"{str(job[0])} \t {job[1]} \t {job[2]} \t {job[3]} \t {job[4]}")
        print("\n")
    return job_results


# COMMAND ----------

userinput = "I am a "
noOfJobs = 5
showjobs(userinput, noOfJobs)

# COMMAND ----------


