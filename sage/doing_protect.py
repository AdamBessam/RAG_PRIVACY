"""
SAGE — doing_protect.py
Adapté depuis : https://github.com/phycholosogy/RAG-SAGE

Implémentation fidèle de :
  "Mitigating the Privacy Issues in RAG via Pure Synthetic Data" (EMNLP 2025)

Modifications par rapport au code original :
  - AzureOpenAI → openai.OpenAI (API standard)
  - Ajout du domaine "echr" (documents juridiques ECHR/CEDH)
  - llm_config autogen adapté pour l'API OpenAI standard
  - Modèles : gpt-3.5-turbo (Stage 1) et gpt-4 (Stage 2 agents), conformes au papier
"""

import os
import re
import json
from tqdm import tqdm
from openai import OpenAI
from autogen import ConversableAgent


def get_llm_client(llm_name: str = 'gpt-3.5-turbo') -> OpenAI:
    """Retourne un client OpenAI standard (remplace AzureOpenAI du code original)."""
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def get_llm_output(prompt: str, llm_client: OpenAI, model_name: str,
                   system_content: str = "You are a helpful assistant.") -> str:
    """Appel LLM avec retry (jusqu'à 8 tentatives), conforme au code original."""
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user",   "content": prompt},
    ]
    output = ''
    for _ in range(8):
        try:
            response = llm_client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=512,
                n=1,
                temperature=0.6,
            )
            output = response.choices[0].message.content
        except Exception:
            output = ''
        if output:
            break
    return output


def get_attributes_prompt(input_context: str, dataset: str) -> str:
    """
    Stage 1a — Extraction des attributs structurels du document.
    Conforme à la fonction get_attributes_prompt du papier.
    Ajout du domaine 'echr' pour les documents juridiques CEDH.
    """
    if dataset.find('chat') != -1:
        prompt = f"""
            Please summarize the key points from the following Doctor-Patient conversation:


            {input_context}

            Provide a summary for the Patient's information, including:
            [Attribute 1: Clear Symptom Description]
            [Attribute 2: Medical History]
            [Attribute 3: Current Concerns]
            [Attribute 4: Recent Events]
            [Attribute 5: Specific Questions]

            Then, provide a summary for the Doctor's information, including:
            [Attribute 1: Clear Diagnosis or Assessment]
            [Attribute 2: Reassurance and Empathy]
            [Attribute 3: Treatment Options and Explanations]
            [Attribute 4: Follow-up and Next Steps]
            [Attribute 5: Education and Prevention]

            Please format your response as follows:

            Patient:
            - [Attribute 1: Clear Symptom Description]:
            - [Attribute 2: Medical History]:
            - [Attribute 3: Current Concerns]:
            - [Attribute 4: Recent Events]:
            - [Attribute 5: Specific Questions]:

            Doctor:
            - [Attribute 1: Clear Diagnosis or Assessment]:
            - [Attribute 2: Reassurance and Empathy]:
            - [Attribute 3: Treatment Options and Explanations]:
            - [Attribute 4: Follow-up and Next Steps]:
            - [Attribute 5: Education and Prevention]:

            Please provide a concise summary for each attribute, capturing the most important information related to that attribute from the conversation.
            """
    elif dataset.find('wiki') != -1:
        prompt = f"""
            Please summarize the key points from the following wiki text:


            {input_context}

            Provide a summary the knowledge from the wiki text, including:
            [Attribute 1: Clear TOPIC or CENTRAL IDEA of the wiki text]
            [Attribute 2: Main details of the TOPIC or CENTRAL IDEA]
            [Attribute 3: Important facts, data, events, or viewpoints]

            Please format your response as follows:

            - [Attribute 1: Clear TOPIC or CENTRAL IDEA of the wiki text]:
            - [Attribute 2: Main details of the TOPIC or CENTRAL IDEA]:
            - [Attribute 3: Important facts, data, events, or viewpoints]:

            Please provide a concise summary for each attribute, capturing the most important information related to that attribute from the conversation. And remember to maintain logical order and accuracy.
            """
    elif dataset.find('echr') != -1:
        # Domaine juridique CEDH — adapté pour ildpil/text-anonymization-benchmark
        prompt = f"""
            Please summarize the key points from the following European Court of Human Rights (ECHR) legal document:


            {input_context}

            Provide a summary of the case, including:
            [Attribute 1: Type of Rights Violation Claimed]
            [Attribute 2: General Profile of the Applicant (occupation, nationality — NO real names)]
            [Attribute 3: Core Facts of the Case (without names, specific addresses or identifiers)]
            [Attribute 4: Procedural History (courts approached, legal steps taken)]
            [Attribute 5: ECHR Articles Alleged to be Violated]
            [Attribute 6: Court's Legal Reasoning and Analysis]
            [Attribute 7: Judgment Outcome]

            Please format your response as follows:

            - [Attribute 1: Type of Rights Violation Claimed]:
            - [Attribute 2: General Profile of the Applicant]:
            - [Attribute 3: Core Facts of the Case]:
            - [Attribute 4: Procedural History]:
            - [Attribute 5: ECHR Articles Alleged to be Violated]:
            - [Attribute 6: Court's Legal Reasoning and Analysis]:
            - [Attribute 7: Judgment Outcome]:

            Please provide a concise summary for each attribute. Do NOT include any real personal names, addresses, or other identifying information in your summary.
            """
    else:
        prompt = 'prompt error'
    return prompt


def get_synthetic_prompt(input_attributes: str, dataset: str) -> str:
    """
    Stage 1b — Génération du document synthétique à partir des attributs.
    Conforme à la fonction get_synthetic_prompt du papier.
    Ajout du domaine 'echr'.
    """
    if dataset.find('wiki') != -1:
        prompt = f"""Here is a summary of the key points:

        {input_attributes}

        Please generate a wiki text using the ALL key points provided.
        The conversation should like a real-word wiki text.
        """
    elif dataset.find('chat') != -1:
        prompt = f"""Here is a summary of the key points:

        {input_attributes}

        Please generate a SINGLE-ROUND patient-doctor medical dialog using the ALL key points provided.
        The conversation should like a real-word medical conversation and contain ONLY ONE question from the patient and ONE response from the doctor. The format should be as follows:

        Patient:[Patient's question contains ALL Patient's key points provided]
        Doctor:[Doctor's response contains ALL Doctor's key points provided]

        Do not generate any additional rounds of dialog beyond the single question and response specified above."""
    elif dataset.find('echr') != -1:
        # Domaine juridique CEDH
        prompt = f"""Here is a summary of the key points from an ECHR legal case:

        {input_attributes}

        Please generate a COMPLETELY FICTIONAL ECHR legal document excerpt using ALL the key points provided.

        CRITICAL RULES:
        1. Use ENTIRELY fictional names for all persons, places, and organizations.
        2. Use fictional but plausible case reference numbers (e.g., "Application No. 12345/99").
        3. The document should read like a realistic ECHR judgment excerpt.
        4. Maintain the same legal structure and terminology as real ECHR documents.
        5. Do NOT copy any real personal information from the original case.

        Generate the synthetic ECHR legal document now:"""
    else:
        prompt = 'prompt error'
    return prompt


def get_synthetic_context(ori_contexts: list, dataset: str,
                           attributes_llm: str = 'gpt-3.5-turbo',
                           synthetic_llm: str = 'gpt-3.5-turbo') -> tuple:
    """
    Stage 1 complet — extraction d'attributs + génération synthétique.
    Conforme à get_synthetic_context du papier.
    Modèles par défaut : gpt-3.5-turbo (papier : gpt-35-turbo).
    """
    attributes_llm_client = get_llm_client(attributes_llm)
    synthetic_llm_client  = get_llm_client(synthetic_llm)

    all_attributes_con = []
    all_synthetic_con  = []

    for ori_context in tqdm(ori_contexts, desc="Stage 1 — generate synthetic context"):
        attributes_con = []
        synthetic_con  = []
        for ori_con in ori_context:
            attributes_prompt  = get_attributes_prompt(ori_con, dataset)
            attributes_context = get_llm_output(attributes_prompt, attributes_llm_client,
                                                attributes_llm, 'You are a helpful assistant.')
            synthetic_prompt   = get_synthetic_prompt(attributes_context, dataset)
            synthetic_context  = get_llm_output(synthetic_prompt, synthetic_llm_client,
                                                synthetic_llm, 'You are a helpful assistant.')
            attributes_con.append(attributes_context)
            synthetic_con.append(synthetic_context)

        all_attributes_con.append(attributes_con)
        all_synthetic_con.append(synthetic_con)

    return all_attributes_con, all_synthetic_con


def get_agent2_context(ori_contexts: list, sync_contexts: list,
                       agents_llm: str = 'gpt-4') -> list:
    """
    Stage 2 — raffinement par agents Joe (privacy) + Cathy (réécriture).
    Conforme à get_agent2_context du papier.
    Modèle par défaut : gpt-4 (conforme au papier pour les agents).

    Adaptation : llm_config pour openai.OpenAI standard (vs AzureOpenAI original).
    """
    openai_api_key = os.getenv("OPENAI_API_KEY")
    llm_config = {
        "config_list": [
            {"model": agents_llm, "api_key": openai_api_key}
        ],
        "temperature": 0.6,
    }

    cathy = ConversableAgent(
        "cathy",
        system_message=(
            "You are a synthetic data generator and your role is to generate synthetic data "
            "based on provided feedback(words after SUGGESTIONS:) and to make sure the synthetic "
            "data is of high utility and privacy-preserving, you should put your generated data "
            "after the word 'GENERATED DATA:' "
        ),
        llm_config=llm_config,
        human_input_mode="NEVER",
        is_termination_msg=lambda msg: "thisissafe" in msg["content"].lower()
    )

    joe = ConversableAgent(
        "joe",
        system_message="""You are a privacy evaluation agent and your role is to provide comprehensive feedback on the synthetic data generated by the synthetic data generator. To be specific, you should analyze the synthetic data (the data after the word 'GENERATED DATA:') from the following aspects:

    1. Personally Identifiable Information (PII): Check if the synthetic data contains any PII, such as names, addresses, phone numbers, email addresses, or other information that can directly identify an individual. If found, suggest ways to remove or anonymize such information.

    2. Sensitive Attributes: Look for any sensitive attributes in the synthetic data, including but not limited to race, ethnicity, religion, political affiliation, sexual orientation, health status, or financial information. If present, provide suggestions on how to handle or obfuscate these attributes to mitigate potential privacy risks.

    3. Contextual Privacy: Evaluate if the synthetic data, when combined with other publicly available information, could potentially lead to the identification of individuals or reveal sensitive information about them. If such risks are identified, recommend strategies to mitigate these contextual privacy issues.

    4. Data Linkage: Assess if the synthetic data can be linked with other datasets to infer additional sensitive information about individuals. If linkage risks are found, suggest techniques such as data perturbation or aggregation to reduce these risks.

    5. Semantic Consistency: Ensure that the privacy-preserving transformations applied to the synthetic data maintain semantic consistency and do not introduce any unintended biases or inaccuracies. If inconsistencies are detected, provide feedback on how to maintain the balance between privacy protection and data utility.

    6. Original Data Recovery: Analyze the synthetic data to determine if it could potentially allow attackers to recover or reconstruct the original conversation data(word behind TRUE CONVERSATION:). If such vulnerabilities are identified, suggest methods to introduce additional randomness, noise, or perturbations to break direct correspondences between the synthetic data and the original conversation, making recovery attempts more difficult.

    Only if the generated data is completely safe and satisfies all the above privacy requirements and prevents the recovery of the original data, include the word 'THISISSAFE' anywhere in your response to signal the end of the evaluation process. Otherwise, provide detailed suggestions and guidance on how to improve the privacy aspects of the synthetic data(after the word "SUGGESTIONS:") and do not contain the word 'THISISSAFE' in your response.

    If the data is deemed safe, please also extract the safe synthetic data (the text after 'GENERATED CONVERSATION:') and return it in the following format:
    SAFE_DATA:
    [BEG]<safe_synthetic_data>[END]THISISSAFE

    Note that your job is only to assess the privacy level of generated data, you can answer either suggestions(SUGGESTIONS) or this data is safe(SAFE_DATA:
    [BEG]<safe_synthetic_data>[END]THISISSAFE), does not provide irrevenlent answers.
    """,
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    joe.reset()
    cathy.reset()

    safe_count     = 0
    safe_data_list = []
    unsafe_lst     = []
    num_turns      = []

    for i in tqdm(range(len(ori_contexts)), desc="Stage 2 — agent refinement"):
        syn_conversation  = sync_contexts[i]
        true_conversation = ori_contexts[i]
        safe_data_lst     = []

        for j in range(len(syn_conversation)):
            syn_con  = syn_conversation[j]
            true_con = true_conversation[j]
            message  = (
                f"Hi Joe, I will give you the real data(TRUE DATA) and synthetic data(GENERATED DATA), "
                f"please help me assess and provide suggestions from the privacy level of "
                f"TRUE DATA:{true_con}\n GENERATED DATA:{syn_con}"
            )

            try:
                result           = cathy.initiate_chat(joe, message=message, max_turns=5)
                safe_data_match  = re.search(r'\[BEG\](.*?)\[END\]',
                                             result.chat_history[-1]['content'], re.DOTALL)
                num_turns.append(len(result.chat_history))
            except Exception:
                result          = 'No'
                safe_data_match = None
                num_turns.append(0)

            if safe_data_match:
                safe_count += 1
                safe_data   = safe_data_match.group(1)
            else:
                if result == 'No':
                    safe_data = syn_con
                elif len(result.chat_history) >= 2:
                    safe_data = result.chat_history[-2]['content'][16:]
                else:
                    safe_data = syn_con
                unsafe_lst.append([i, j])

            safe_data_lst.append(safe_data)
            joe.reset()
            cathy.reset()

        safe_data_list.append(safe_data_lst)

    print(f'Number of safe data: {safe_count}')
    print(f'Unsafe items: {unsafe_lst}')
    if num_turns:
        avg_turns = sum(num_turns) / len(num_turns)
        print(f'Average turns (Stage 2): {avg_turns:.2f}  (paper: 3.964)')

    return safe_data_list
