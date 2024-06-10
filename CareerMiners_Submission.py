# Databricks notebook source
!pip install mlflow[genai]>=2.9.0
!pip install databricks-vectorsearch

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import mlflow.deployments
import json
import os
from databricks.vector_search.client import VectorSearchClient



# COMMAND ----------

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

def getUserSkill(userinput):
    response = getLLMResponse(userinput)
    skills = json.loads(response)
    print(f'response from LLM, {skills}')
    skills_list = skills['skills']
    skill_set  =  ', '.join(skills_list)
    print(f'Skills Required : {skill_set}')
    return skill_set


# COMMAND ----------

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

def showjobs(userinput, noOfJobs):
    job_results = getJobs(userinput, noOfJobs)
    print("Top 5 matches on the user query:")
    print("\n\n----------------------------------------------")
    print("\nJOB_ID \t Title \t Company \t Job Description \t Relevancy Score")
    print("\n----------------------------------------------")
    for job in job_results:
        #column_index = job[0].index("job_id")
        print(f"{job[0]} \t {job[1]} \t {job[2]} \t {job[3]} \t {job[4]}")
        print("\n")
    return job_results


# COMMAND ----------

userinput = "I am a business analyst with Databricks experience looking for a job."
noOfJobs = 5
showjobs(userinput, noOfJobs)

# COMMAND ----------


